# DotCache Layer-Page Selection Revision Tracker

This tracker turns the March 2026 review into concrete paper and experiment work. It assumes the working manuscript source is now [`dotcache_layer_page_selection.md`](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md).

## Status

- latest `main` merged into the working branch on `2026-03-31`
- reviewed PDF artifact pulled into repo as editable markdown
- first manuscript cleanup pass completed:
  - replaced the false monotonic-tier story with policy-bundle language
  - separated `exact` policy labels from `M3 escape`
  - softened the sub-linear decode claim to selected-set attention only
  - added selector-overhead evidence
  - added missing related-work positioning
  - removed table placeholders like `see below`, `see text`, `varies`, and `(est.)`

## Review Issue -> Fix

### 1. Empirical section is not publication-grade

Paper fix:

- keep the current evidence section honest and explicitly prototype-grade
- define `loss delta`, `token agreement`, `replay_output_max_abs_error`, and `teacher_forced_logit_max_abs_error`
- stop mixing incomparable metrics in one summary table

Experiment work still needed:

- one held-out benchmark suite with fixed prompt counts and variance
- one hardware table with exact GPU / backend / dtype / batch size
- one standard long-context suite: `LongBench`, `RULER`, or `Needle-in-a-Haystack`

Useful existing artifacts:

- [`docs/benchmark_report.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/benchmark_report.md)
- [`docs/local_layer_profiles.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/local_layer_profiles.md)
- [`docs/qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md)

### 2. Tier abstraction has cracks

Paper fix:

- rename the concept from "sensitivity tiers" to "policy bundles"
- show K and V candidate order explicitly
- state that bundle labels are implementation shorthands, not a universal conservatism ladder

Code anchor:

- [`planner.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/planner.py)

### 3. `exact` is ambiguous

Paper fix:

- reserve `M3 escape` for explicit high-precision storage
- reserve "exact baseline" for no-approximation runtime runs
- use `exact` only as the bundle label meaning "do not adaptively search alternate candidates"

### 4. Sub-linear decode claim is under-argued

Paper fix:

- change the claim to: "selected-set attention can scale sub-linearly while the current selector pass remains linear in candidate pages"
- cite the replay microbench instead of implying the selector is free

Existing evidence:

- [`qwen35_selector_replay_microbench_smoke.json`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_selector_replay_microbench_smoke.json)

Still needed:

- end-to-end selector cost inside the serving loop
- selector + attention + value-mix breakdown at multiple contexts

### 5. Too many codec modes for the amount of validation

Paper fix:

- keep all six modes in the format table because they exist in the page envelope
- explicitly say the current empirical story is mainly about `M0`, `M1`, `M2`, and `M3`
- describe `M4` and `T3` as supported but still lightly validated

Potential future split:

- if `M4` and `T3` stay thinly evaluated, move them to an extension section in the next draft

### 6. Page-level adaptivity is weaker than layer-level heuristics

Paper fix:

- say this directly
- frame Qwen2.5 selective exact-K as the strongest current page-level or layer-head rescue result
- avoid claiming that per-page observed statistics already beat simpler per-layer policies

Still needed:

- fixed per-layer policy vs adaptive per-page routing ablation
- fixed shortlist vs page-stat-conditioned shortlist ablation

### 7. Manual profile discovery is a reproducibility risk

Paper fix:

- call profiles "hand-authored current bundles" instead of "validated final policies"
- move automated discovery from a vague future-work line into a concrete next-step requirement

Existing starting point:

- [`scripts/distill_qwen35_serving_shortlist_rule.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/distill_qwen35_serving_shortlist_rule.py)
- [`offline_distillation_layer23_mps_cuda_v2.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/offline_distillation_layer23_mps_cuda_v2.md)

Still needed:

- calibration/test split for profile search
- frozen-policy held-out evaluation

## High-Value Next Experiments

### A. Standard Qwen3.5 shortlist table

Goal:

- produce one clean table with `TTFT`, decode `ms/step`, `tok/s`, selected pages, candidate pages, and quality error at `4096`, `8192`, `16384`

Starting point:

- [`docs/qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md)

Command:

```bash
PATH=/Users/deanocalver/Documents/Projects/DotCache/.venv/bin:$PATH \
PYTHONPATH=/Users/deanocalver/Documents/Projects/DotCache \
.venv/bin/python scripts/run_qwen35_cuda_shortlist_probe.py \
  --contexts 4096 8192 16384 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --profile-backend \
  --output benchmarks/results/qwen35_cuda_shortlist_probe.jsonl
```

### B. Page-level adaptivity ablation

Goal:

- compare:
  - fixed per-layer bundle
  - fixed per-layer plus explicit rescue overrides
  - current per-page observed-stat routing

Minimal models:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

### C. Selector overhead in the real loop

Goal:

- report selector compute time as a fraction of decode time, not just as an isolated replay microbench

Needed outputs:

- selector compute
- shortlist materialization
- score kernel
- mix kernel
- softmax
- total decode

### D. Calibration/Test profile search

Goal:

- stop hand-tuning until the plots look nice

Minimum viable version:

- pick a fixed calibration prompt set per model
- search bundle assignments or rescue rules on calibration only
- freeze the policy
- evaluate once on held-out prompts

## Mainline Results Pulled From Other Machines

Latest `main` added fresh ROCm 890M StateCache artifacts and docs:

- [`docs/performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md)
- [`benchmarks/results/qwen35_rocm_890m_statecache_discovery_20260331/`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_rocm_890m_statecache_discovery_20260331)

These do not directly strengthen the page-selection paper, but they are now merged locally and available if we later broaden the paper's systems comparison section.

## Suggested Order

1. Freeze the manuscript wording cleanup already started in [`dotcache_layer_page_selection.md`](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md).
2. Run one clean Qwen3.5 shortlist table with end-to-end timings and error metrics.
3. Run one explicit page-routing ablation against fixed per-layer policies.
4. Only after that, decide whether the paper is still a workshop/system-note submission or ready to target a stronger venue.
