# DotCache Submission Execution Plan

This plan converts the current review feedback and March 31 protocol work into an ordered submission path. It assumes the working paper source is [dotcache_layer_page_selection.md](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md), the evaluation contract is [dotcache_page_selection_standardized_evaluation.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_page_selection_standardized_evaluation.md), and the next-stage decision framework is [dotcache_compressed_page_test_readiness_rfc.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_compressed_page_test_readiness_rfc.md).

## Current Starting Point

What is already true:

- the Qwen3.5 CUDA shortlist lane now has protocol-tagged `held_out` quality rows
- the large-context systems repro lane now reports `n`, mean, min, max, stddev, and `95% CI`
- grouped CUDA is no longer broken, but it is not the default winner
- the `49152` quality regime is still not clean

Key current artifacts:

- [qwen35_cuda_shortlist_large_context_repro_serving_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving_summary.md)
- [qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl)
- [qwen35_cuda_shortlist_paper_table.md](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_paper_table.md)

## Submission Goal

Reach the minimum evidence bar for a credible workshop paper or strong systems preprint:

- page-format selection is evaluated as the main experiment, not a support heuristic
- the four-lane contract is frozen and used consistently
- one oracle or trace-backed proof shows heterogeneous page formats beat the best fixed single format
- matched-budget external baselines exist on one strong quality model and one systems model
- runtime truth includes `TTFT`, `p95` decode latency, and effective-memory accounting
- the benchmark suite includes at least one hard realistic long-context family, one controlled stress family, and one decode-heavy reasoning slice
- the current benchmark misses are explained through a page-level failure workbook

## Ordered Workstreams

### Phase 1. Freeze The Evaluation Contract And Logging Schema

Goal:

- make every next-stage run conform to one contract before changing the selector or mode bundles again

Tasks:

1. require the four-lane tags on every promoted run
2. require `TTFT`, `p95` decode latency, effective bytes per token, page-format histograms, and selector timing splits
3. add a hard distinction between `reference_trace` and `paged_runtime`
4. define one matched-budget accounting rule that counts metadata and fragmentation

Required outputs:

- updated [dotcache_page_selection_standardized_evaluation.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_page_selection_standardized_evaluation.md)
- one shared logging schema note in `docs/` or `benchmarks/`

Success criteria:

- new rows are promotable without ad hoc explanation
- future baseline wiring has one budget rule instead of per-method caveats

### Phase 2. Build The Full-Precision Trace Recorder And Oracle Replay Harness

Goal:

- promote page-format determination from a heuristic bundle story to a measurable oracle problem

Tasks:

1. capture full-precision traces for a fixed model roster
2. replay each page across the current candidate menu:
   - `M0-2b`
   - `M0-3b`
   - `M0-4b`
   - `M1-4b`
   - `M2-4b`
   - `M4-4b`
   - `M3 exact`
   - `T3` when stable
3. label each page with the cheapest safe format under frozen fidelity thresholds
4. compare:
   - current heuristics
   - a static layer or head or age map
   - a lightweight learned predictor

Required outputs:

- trace-capture entrypoint in `scripts/` or `benchmarks/`
- replay harness that emits per-page oracle labels
- one summary note describing thresholds and freeze rules

Success criteria:

- DotCache can answer whether a heterogeneous menu beats the best fixed single format on oracle traces

### Phase 3. Sweep The Existing DotCache Menu Before Inventing New Policies

Goal:

- establish the true shape of the current page menu over page size, mode, and recent-window choices

Tasks:

1. run the current menu across page-size sweeps
2. compare fixed single-format variants against the heterogeneous menu
3. keep Qwen3.5 0.8B as the wind-tunnel model for selector stress and shortlist behavior
4. add at least one stronger Qwen or Llama-family model for final quality verdicts

Required outputs:

- one fixed-format versus heterogeneous summary table
- one strong-model held-out quality table

Success criteria:

- the next stage stops treating Qwen3.5 alone as the final quality courtroom

### Phase 4. Add Wave-1 External Baselines Under Matched Effective Budgets

Goal:

- test DotCache against the canonical external families before spending more time on internal heuristic tuning

Mandatory first-wave baselines:

- compression-side:
  - `KIVI`
  - `KVQuant`
  - `QJL`
  - `QServe`
  - `InnerQ`
  - `RotateKV`
  - `PM-KVQ`
  - one saliency-aware representative such as `ZipCache` or `MiKV`
- read-time comparators:
  - `Quest`
  - `H2O`
  - `SnapKV`
  - `PQCache`
  - StreamingLLM sink-plus-recent window

Stretch additions:

- `Expected Attention`
- `ParisKV`
- `Self-Indexing KVCache`

Required outputs:

- matched-budget comparison notes
- one systems-model baseline table
- one stronger-model quality baseline table

Success criteria:

- DotCache beats at least part of the canonical set on one strong quality model and one systems model, or the paper wording is narrowed accordingly

### Phase 5. Run The Hard Benchmark Suite And Build The Failure Workbook

Goal:

- replace flattering narrow slices with a small benchmark suite that is difficult to game

Near-term suite:

- controlled stress:
  - `RULER`
  - `NoLiMa`
- realistic long-context understanding:
  - `LongBench v2`
  - `LongBench Pro`
  - `InfinityBench`
- decode-heavy reasoning:
  - `GSM8K`
  - `MATH500`
- deployment gate:
  - one multi-instruction and prompt-leakage pack

Tasks:

1. run the suite under the frozen contract
2. keep page-level miss diagnostics for every failed answer
3. classify each miss as:
   - pruned by read-time selection
   - kept but damaged by write-time format
   - present and healthy but still under-attended

Required outputs:

- benchmark summaries under the standardized table shape
- a failure workbook artifact for every promoted benchmark family

Success criteria:

- the benchmark-quality pain is reduced to concrete selector, format, or budget failures rather than vague regressions

### Phase 6. Promote Only Selectors That Survive The Bakeoff

Goal:

- keep the serving path honest by promoting only selectors that win under the matched-budget and benchmark contract

Tasks:

1. compare oracle-closeness, held-out quality, and held-out systems results
2. reject selectors that only improve Needle or passkey but fail the hard suite
3. require no `TTFT` cliff, no `p95` disaster, and no obvious leakage or instruction-following regression

Second-wave baselines to add once the canonical set is in place:

- `TailorKV`
- `Kitty`
- `DiffKV`

Success criteria:

- only selectors that survive the matched-budget bakeoff graduate into the real serving path

## Recommended Execution Order

1. Phase 1: freeze the evaluation contract and logging schema
2. Phase 2: build the full-precision trace recorder and oracle replay harness
3. Phase 3: sweep the current DotCache menu across page sizes and fixed-format comparisons
4. Phase 4: add wave-1 external baselines under matched effective budgets
5. Phase 5: run the benchmark suite and generate the page-level failure workbook
6. Phase 6: promote only selectors that survive the bakeoff

## What Counts As Submission-Ready

Minimum bar for a workshop/system-note submission:

- protocolized systems and quality tables with variance
- one oracle-backed proof that heterogeneous page formats beat the best fixed single format
- one strong-model quality verdict beyond Qwen3.5 0.8B
- matched-budget external baselines
- `TTFT` and `p95` runtime truth
- a benchmark-backed explanation of the current quality failures

Stronger bar for a more ambitious venue:

- all of the above
- both an external baseline and a 7B+ standardized point
- minimal automated profile search
- selector timing decomposition in the main systems section

## Immediate Next Action

The next concrete move should be:

1. lock the shared logging schema around the expanded evaluation contract
2. define the first full-precision trace recorder plus oracle-replay harness surface

That path creates the foundation for every later baseline, benchmark, and selector bakeoff.
