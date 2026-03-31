# DotCache Submission Execution Plan

This plan converts the current review feedback and March 31 protocol work into an ordered submission path. It assumes the working paper source is [dotcache_layer_page_selection.md](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md) and the evaluation contract is [dotcache_page_selection_standardized_evaluation.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_page_selection_standardized_evaluation.md).

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

- one standardized external task
- one external baseline on the Qwen3.5 lane
- variance-aware main tables
- a clean stance on the `49152` quality story
- one 7B+ standardized data point
- minimal scientific profile search
- selector timing decomposition inside the real serving loop

## Ordered Workstreams

### Phase 1. Lock the Current Qwen3.5 Tables

Goal:

- upgrade the current Qwen3.5 CUDA tables from one-off probe summaries to protocol-compliant paper tables

Tasks:

1. fold the repro summary into the manuscript tables for the `32768/49152` systems rows
2. label the `49152` quality rows explicitly as `held_out` and synthetic prompt family
3. report `n=3`, mean, and `95% CI` for the current systems rows
4. keep the current claim narrow until a better `49152` rescue exists

Inputs:

- [qwen35_cuda_shortlist_large_context_repro_serving_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving_summary.md)
- [qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl)

Deliverables:

- updated [qwen35_cuda_shortlist_paper_table.md](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_paper_table.md)
- updated [dotcache_layer_page_selection.md](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md)

Done when:

- Tables 6 and 7 can be populated from committed summary artifacts rather than hand-picked journal prose

### Phase 2. Add One Named Benchmark Task

Goal:

- add the cheapest recognizable external task under the standardized protocol

Chosen first task:

- Needle-in-a-Haystack

Why first:

- cheapest wiring cost
- directly relevant to long-context retrieval/selectivity claims
- gives the paper one named benchmark without waiting for a full LongBench or RULER integration

Tasks:

1. add a protocol-tagged Needle harness for the Qwen3.5 attention-subset lane
   Status: wired via [bench_qwen35_attention_subset_dotcache_needle.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_needle.py), [run_qwen35_cuda_needle_probe.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_needle_probe.py), and [run_qwen35_cuda_needle_protocol.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_needle_protocol.sh)
2. run exact, shortlist base, and shortlist `layer:23` variants
3. record both retrieval success and systems metrics under the same prompt suite metadata

Required outputs:

- `benchmarks/results/qwen35_cuda_needle_*.jsonl`
- `docs/qwen35_cuda_needle_summary.md`

Success criteria:

- at least one named task appears in the paper with prompt count, context, and variance

Decision gate:

- if Needle is positive, it becomes the first external task result in the paper
- if Needle is mixed or negative, it still improves credibility and informs claim narrowing

### Phase 3. Add One External Baseline On The Qwen3.5 Lane

Goal:

- compare DotCache against one non-internal long-context method under a matched memory budget

Baseline candidates:

- `Quest`
- `H2O`

Selection rule:

- choose whichever can be wired fastest with honest budget matching and reproducible runs
- do not block this phase waiting for the theoretically best baseline if a simpler one is practical now

Tasks:

1. pick one baseline and define a memory-budget matching rule
2. run it on the Qwen3.5 lane at the same main contexts used in the paper
3. compare against exact and shortlist base under the standardized systems table shape

Required outputs:

- `benchmarks/results/qwen35_cuda_external_baseline_*.jsonl`
- `docs/qwen35_cuda_external_baseline_summary.md`

Success criteria:

- one external comparator appears in the paper with matched budget and explicit caveats

### Phase 4. Resolve The `49152` Quality Fork

Goal:

- stop carrying an unresolved half-claim into submission

Two acceptable outcomes:

1. `49152` quality rescue works well enough to promote
2. the paper is explicitly reframed as a systems-forward paper with unresolved hardest-regime quality

Tasks:

1. run one stronger `held_out quality` rescue beyond `top_k=8`
2. compare it directly against the current protocol-tagged `49152` rows
3. make an explicit go/no-go decision on the quality claim

Current baseline to beat:

- `shortlist_base`: `loss delta 0.0130062`, `max logit abs error 7.0`
- `shortlist_l23_ctx`: `loss delta 0.0128626`, `max logit abs error 6.96484375`

Required outputs:

- `benchmarks/results/qwen35_cuda_shortlist_49152_*_heldout_quality.jsonl`
- one short decision note in [performance_journal.md](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md)

Decision gate:

- if no rescue clearly improves the `49152` held-out lane, freeze the paper wording as systems-forward and stop spending cycles pretending a quality fix is imminent

### Phase 5. Add One 7B+ Standardized Point

Goal:

- show that the story is not limited to a tiny hybrid model

Preferred first target:

- Qwen2.5 7B, because the repo already has relevant CUDA work and quality references

Tasks:

1. choose one standardized context bucket
2. run one `held_out systems` row and one `held_out quality` row
3. keep the case count small: exact plus one DotCache operating point

Required outputs:

- `benchmarks/results/qwen25_7b_*_standardized_*.jsonl`
- `docs/qwen25_7b_standardized_summary.md`

Success criteria:

- one 7B+ table row appears under the same reporting contract as Qwen3.5

### Phase 6. Minimal Automated Profile Search

Goal:

- remove the most obvious “hand-tuned until it worked” criticism

Minimum viable version:

1. freeze a calibration prompt set
2. search a small policy space only on calibration
3. emit a frozen profile
4. evaluate once on held-out

Required outputs:

- `scripts/` or `benchmarks/` entrypoint for profile search
- committed calibration split definition
- one `docs/` note describing search space and freeze rule

Success criteria:

- the manuscript can say profile search was separated from held-out evaluation, even if the search itself is still simple

### Phase 7. Selector Timing Decomposition In The Real Serving Loop

Goal:

- explain the remaining shortlist and grouped-path tradeoffs with end-to-end evidence

Tasks:

1. surface selector / ranking / materialization / backend timing totals in the promoted serving rows
2. compare default shortlist versus grouped shortlist on the same contexts
3. use this as diagnostic support, not as a substitute for the main systems table

Required outputs:

- `benchmarks/results/qwen35_cuda_selector_timing_*.jsonl`
- `docs/qwen35_cuda_selector_timing_summary.md`

Success criteria:

- the paper can support its selector-overhead claims with real serving-loop timings rather than only replay microbenchmarks

## Recommended Execution Order

1. Phase 1: lock the current tables
2. Phase 2: Needle-in-a-Haystack
3. Phase 4: resolve the `49152` quality fork
4. Phase 7: selector timing decomposition
5. Phase 3: one external baseline
6. Phase 5: one 7B+ standardized point
7. Phase 6: minimal automated profile search

## What Counts As Submission-Ready

Minimum bar for a workshop/system-note submission:

- protocolized Qwen3.5 tables with variance
- one named benchmark task
- one explicit stance on the `49152` quality issue
- one external baseline or one 7B+ standardized point

Stronger bar for a more ambitious venue:

- all of the above
- both an external baseline and a 7B+ standardized point
- minimal automated profile search
- selector timing decomposition in the main systems section

## Immediate Next Action

The next concrete move should be:

1. update the manuscript tables from the new repro summary
2. wire Needle-in-a-Haystack under the standardized protocol

That path gives the fastest improvement in credibility per unit effort.
