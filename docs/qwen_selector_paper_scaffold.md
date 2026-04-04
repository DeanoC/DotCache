# Qwen Selector Paper Scaffold

This file is a working markdown scaffold for the paper draft we want to co-write with ChatGPT Pro.

It is intentionally:

- manuscript-shaped rather than journal-shaped
- conservative about claims
- explicit about what is already supported versus what is still waiting on the in-flight matrix

The goal is to let us fill in numbers and tighten wording as new results arrive, without having to redesign the paper structure each time.

## Working Title Options

1. `Learned Page-Format Selection for Compressed-Domain KV-Cache Execution`
2. `Selector-Guided Compressed KV Execution for Long-Context Serving`
3. `Family-Sensitive Page Selection for DotCache Long-Context Inference`

## One-Sentence Thesis

DotCache can execute attention directly on compressed KV pages, and a small learned page-format selector makes that execution fast enough to be the default serving path on Qwen without sacrificing the task and held-out quality rows we currently trust.

## Current Claim Boundary

This is the strongest current claim we should be comfortable making before the new matrix finishes:

- On Qwen3.5, the `systems` learned-selector profile is the right default serving path for the current DotCache runtime.
- That claim holds across compact held-out task checks, a small LongBench-derived QA pack, and native serving/backend-truth runs up through Qwen3.5 27B.
- On Llama 3.2 3B, the learned selector is already saturated to `M3`, so `quality` and `systems` are effectively the same operating point.
- The remaining bottleneck is backend `score + mix` cost on the M3-heavy path, not selector overhead.

What we should not claim yet:

- that DotCache has already beaten every matched-budget external baseline
- that the current LongBench mini-pack is the full long-context quality story
- that one selector bias is globally optimal across all families

## Current Trusted Evidence

These are the result families that are stable enough to cite in the draft now.

### Qwen 9B Task / Profile Check

Primary artifacts:

- [selector_quality_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_selector_quality_compare_20260402/selector_quality_compare.md)
- [task_selector_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_task_selector_compare_20260402_reasoning_v2/task_selector_compare.md)
- [task_selector_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_task_selector_compare_20260402_long_context/task_selector_compare.md)

Current read:

- `systems` preserves success on the compact instruction / retrieval / reasoning slices
- `systems` materially outperforms `quality` on decode
- `systems` is slightly better or equal on the internal quality metrics we already expose

### Qwen 9B LongBench Mini-Pack

Primary artifact:

- [longbench_selector_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_longbench_selector_compare_20260404/longbench_selector_compare.md)

Current read:

- `systems`, `quality`, and `exact` tie on QA F1 on the current mini-pack
- `systems` is much faster than `quality`, `exact`, and the sink-plus-recent streaming reference
- this is a positive held-out external-style checkpoint, but still a mini-pack rather than a full benchmark courtroom

### Qwen 27B Native Backend Truth

Primary artifact:

- [backend_truth_report.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_27b_backend_truth_20260404_native/backend_truth_report.md)

Current read:

- learned selector remains the clear decode winner over exact and shortlist
- this has now been revalidated on the intended newer `transformers 5.5.0` stack
- the scale-up story is therefore no longer just a transferred-artifact or old-environment curiosity

### Cross-Family Checkpoint

Primary artifact:

- [selector_profile_promotion_checkpoint.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/selector_profile_promotion_checkpoint_20260402/selector_profile_promotion_checkpoint.md)

Current read:

- Qwen: promote `systems`
- Llama: `quality` and `systems` are effectively equivalent today

## Results Still In Flight

The new matrix runner is intended to fill these rows:

- Qwen model-size sweep: `4B`, `9B`, `27B`
- benchmark families:
  - compact task compare
  - LongBench compare
  - backend truth
- context slices:
  - task compare: `1024`, `2048`
  - LongBench: `4096`, `8192`
  - backend truth: `1024`, `2048`

Do not hard-code any conclusion from this matrix until the report lands:

- `benchmarks/results/qwen_results_matrix_20260404/qwen_results_matrix.md`

## Draft Abstract

Use this as a starting point, not final copy.

> Long-context serving systems increasingly rely on compressed KV caches, but most prior work treats compression as a storage problem and leaves decode-time execution on dequantized tensors. DotCache instead executes attention directly on compressed KV pages, which turns page-format choice into a runtime systems decision rather than a preprocessing detail. We study learned page-format selection for this setting and show that a small heterogeneous menu can become the default serving path on Qwen-family models. Across held-out task slices, a LongBench-derived QA mini-pack, and native serving/backend-truth runs up to Qwen3.5 27B, the systems-tuned learned selector preserves the task and quality rows we currently trust while substantially reducing decode latency relative to exact and shortlist-style baselines. We further show that selector overhead is small and stable, and that the remaining bottleneck lies in backend score/mix execution on the M3-heavy path rather than in the selector itself. These results suggest that compressed-domain execution is practical when page-format choice is treated as a learned, family-sensitive systems policy rather than a static codec decision.

## Draft Introduction Skeleton

### Problem

- Long-context inference is bottlenecked by KV-cache bandwidth and execution cost.
- Compression helps memory footprint, but it does not automatically solve decode-time execution.
- If the runtime consumes compressed pages directly, page format becomes part of the serving algorithm.

### Gap In Prior Framing

- Many KV-compression papers can treat codec choice as mostly static.
- DotCache cannot: the attention kernel reads compressed pages directly.
- That means bad page choices propagate into the execution path rather than being normalized away by dequantization later.

### Core Idea

- maintain a small page-format menu
- learn which pages should stay cheap, which should escape to high fidelity, and when
- split selector operating points into:
  - `quality`
  - `systems`

### Main Contributions

Placeholder list, to be tightened once the matrix lands:

1. A runtime learned selector for compressed-domain page-format choice in DotCache.
2. A serving integration that uses the learned selector with low overhead and zero observed runtime fallbacks on the validated Qwen lanes.
3. Evidence that the Qwen `systems` selector can be the default serving path on the current runtime without harming the held-out task and LongBench-derived rows we trust.
4. A cross-family result showing that selector policy is family-sensitive: Qwen benefits from an explicit systems bias, while Llama does not currently need one.

## Paper Structure

### 1. Introduction

Write this to motivate the systems question:

- compressed-domain execution makes page selection part of the runtime
- selector quality matters because the kernel consumes compressed pages directly
- we care about both latency and quality retention

### 2. DotCache Execution Model

Subsections:

1. page formats (`M0`, `M3`, etc.)
2. compressed-domain `score` and `mix`
3. write-time page-format selection
4. read-time execution on compressed pages

### 3. Learned Page-Format Selection

Subsections:

1. oracle capture and label generation
2. selector dataset construction
3. `linear_softmax` selector
4. profile split:
   - `quality`
   - `systems`
5. family-sensitive profile resolution

### 4. Experimental Protocol

This section should be strict.

#### 4.1 Models

Placeholder:

| family | models | role in paper |
| --- | --- | --- |
| Qwen3.5 | `4B`, `9B`, `27B` | main scale sweep |
| Llama 3.2 | `3B` | family-sensitivity check |

#### 4.2 Bench Families

Placeholder:

| family | purpose | status |
| --- | --- | --- |
| compact task compare | task-level sanity and promotion gate | ready |
| LongBench QA mini-pack | small external-style held-out check | ready |
| backend truth | serving decomposition and speed truth | ready |
| broader Qwen matrix | scale/context consistency | in flight |

#### 4.3 Baselines

Current baseline vocabulary:

- `exact`
- `quality`
- `systems`
- `streaming_sink_recent`
- `shortlist_base`

Possible future external rows:

- Quest-like matched-budget row
- PQCache-like matched-budget row

#### 4.4 Metrics

Main metrics:

- exact-match task success
- QA F1
- decode ms/step
- p95 decode ms/step
- resident / KV memory

Supporting metrics:

- teacher-forced perplexity ratio
- logit RMSE
- selector agreement / oracle gap
- M3 fraction
- selector microseconds per invocation

## Main Results Section Layout

This is the section we should be able to fill row by row as the matrix finishes.

### 5.1 Main Qwen Matrix

Placeholder summary sentence:

> Across Qwen model sizes and contexts, the learned `systems` profile [PLACEHOLDER: dominate / mostly dominate / cleanly improve] the `quality` profile on decode while [PLACEHOLDER: preserving / mostly preserving] the task and LongBench rows we trust.

#### Table A. Compact Task Matrix

Fill from the new matrix report.

| model | task | context | exact success | quality success | systems success | quality decode ms/step | systems decode ms/step | systems vs quality | quality RMSE | systems RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B` | `instruction` | `1024` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-4B` | `reasoning` | `2048` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `instruction` | `1024` | `1.000` | `1.000` | `1.000` | `129.97` | `42.84` | `3.03x` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `retrieval` | `2048` | `1.000` | `1.000` | `1.000` | `230.65` | `59.22` | `3.89x` | `TBD` | `TBD` |
| `Qwen3.5-27B` | `instruction` | `1024` | `1.000` | `1.000` | `1.000` | `TBD` | `TBD` | `~3x-4x` | `TBD` | `TBD` |
| `Qwen3.5-27B` | `retrieval` | `2048` | `shared fail or TBD` | `shared fail or TBD` | `shared fail or TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

#### Table B. LongBench Matrix

| model | context cap | case | exact match | QA F1 | decode ms/step | p95 decode | ppl ratio | RMSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B` | `4096` | `exact` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-4B` | `4096` | `systems` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `4096` | `exact` | `0.250` | `0.441` | `627.28` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `4096` | `quality` | `0.250` | `0.441` | `583.62` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `4096` | `systems` | `0.250` | `0.441` | `93.73` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `4096` | `streaming` | `0.250` | `0.441` | `257.16` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `8192` | `systems` | `0.000` | `0.291` | `159.38` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-27B` | `4096` | `systems` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

#### Table C. Backend Truth Matrix

| model | context | exact decode ms/step | shortlist decode ms/step | learned decode ms/step | learned vs exact | learned vs shortlist | learned M3 frac | selector us/inv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen3.5-4B` | `1024` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `1024` | `239.43` or latest row | `241.46` or latest row | `74.73` or latest row | `TBD` | `TBD` | `~0.99` | `~25` |
| `Qwen3.5-9B` | `4096` | `796.45` | `345.04` | `180.37` | `4.42x` | `1.91x` | `TBD` | `TBD` |
| `Qwen3.5-9B` | `8192` | `1476.05` | `367.03` | `308.69` | `4.78x` | `1.19x` | `TBD` | `TBD` |
| `Qwen3.5-27B` | `1024` | `507.44` | `524.77` | `169.09` | `3.00x` | `3.10x` | `0.995` | `24.57` |
| `Qwen3.5-27B` | `2048` | `824.98` | `642.79` | `224.14` | `3.68x` | `2.87x` | `0.995` | `24.76` |

### 5.2 Cross-Family Selector Policy

This section should be simple and disciplined.

Current story:

- Qwen benefits from an explicit `systems` bias.
- Llama does not currently need one.

Placeholder table:

| family | model | exact success | quality success | systems success | mean systems vs quality speedup | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen | `Qwen3.5-9B` | `1.000` | `1.000` | `1.000` | `3.679x` | promote `systems` |
| Llama | `Llama-3.2-3B` | `1.000` | `1.000` | `1.000` | `0.981x` | equivalent operating points |

### 5.3 Selector Overhead And Backend Decomposition

This section is where we make the execution-format thesis visible.

Current trusted points:

- selector overhead is small and stable, around `25-27 us/inv`
- selector work is prefill-side, not decode-side
- `unpack` is no longer the dominant learned-lane tax after the tiny-chunk cache fix
- remaining dominant cost is `score + mix` on the M3-heavy path

Placeholder table:

| model | context | selector us/inv | unpack ms/step | score ms/step | mix ms/step | payload read MiB/step | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Qwen3.5-9B` | `1024` | `~26` | `0.00` | `TBD` | `TBD` | `TBD` | learned path dominated by score+mix |
| `Qwen3.5-9B` | `2048` | `~27` | `0.00` | `TBD` | `TBD` | `TBD` | same pattern at longer context |
| `Qwen3.5-27B` | `1024` | `24.57` | `0.00` | `42.32` | `36.77` | `TBD` | same scale story on native larger model |
| `Qwen3.5-27B` | `2048` | `24.76` | `0.00` | `77.85` | `66.10` | `TBD` | backend, not selector, is the remaining bottleneck |

## Discussion Section Layout

### 6.1 What The Selector Is Actually Doing

Possible framing:

- It is not discovering a large codec zoo.
- It is mainly learning when to promote pages into the higher-fidelity execution mode that pays off at serving time.
- On Qwen, the `systems` operating point is strongly M3-heavy.
- On Llama, the unbiased selector already lands there.

### 6.2 Why This Is A Systems Result, Not Just A Classifier Result

- the selector matters because page format changes the decode kernel path itself
- it changes score/mix cost, not just storage footprint
- the selector-to-backend linkage is the actual mechanism

### 6.3 Limits

Keep this candid:

- external matched-budget baselines are still thin
- LongBench mini-pack is not a full suite
- broader context and benchmark coverage are still being built
- backend score/mix is still the dominant remaining systems tax

## Figure Plan

These are the figures I’d expect to make from the results.

### Figure 1. Page-Format Selection Pipeline

Diagram:

- trace capture
- oracle labels
- selector training
- runtime artifact
- serving path

### Figure 2. Qwen Speedup Heatmap

Axes:

- x: context
- y: model size
- value: `systems vs quality` speedup

### Figure 3. Quality Retention Heatmap

Axes:

- x: context
- y: benchmark family / task
- value: `systems - quality` metric delta

### Figure 4. Backend Cost Breakdown

Grouped bars:

- exact
- shortlist
- learned

Components:

- score
- mix
- unpack
- softmax
- chunk assembly

### Figure 5. Selector Mechanism Figure

One compact panel showing:

- M3 fraction
- selector agreement / oracle gap
- task quality
- decode speed

This is the “systems + decision layer insight” figure.

## ChatGPT Pro Collaboration Prompts

These are useful prompts to hand to ChatGPT Pro once this scaffold and the fresh matrix are available.

### Prompt 1. Tighten The Abstract

> Here is the current paper scaffold and the current trusted result set. Rewrite the abstract into a conference-paper style abstract that is conservative, concrete, and novelty-focused. Do not oversell external baseline coverage.

### Prompt 2. Write Introduction Variants

> Using this scaffold, draft three introduction variants:
> 1. systems-heavy
> 2. compression-heavy
> 3. learned-decision-layer-heavy
> Keep them all honest to the current evidence boundary.

### Prompt 3. Turn Results Into Claim Language

> Read the result tables and propose exact phrasing for:
> - the main claim
> - one secondary claim
> - one explicit non-claim
> The language should be publication-safe and should distinguish internal evidence from external-benchmark evidence.

### Prompt 4. Help Build The Discussion

> Based on these results, write a discussion section that explains:
> - why Qwen benefits from a systems selector bias
> - why Llama does not currently need one
> - why selector overhead is not the main bottleneck
> - what is still missing before a broader benchmark claim

## Fill-In Checklist

- [ ] Replace all `TBD` cells with final matrix values
- [ ] Decide final title
- [ ] Tighten abstract
- [ ] Tighten introduction
- [ ] Add exact artifact references for every main table
- [ ] Add one stronger external matched-budget baseline if it lands in time
- [ ] Add selector-to-task linkage figure
- [ ] Add broader LongBench statement only if the broader pack actually lands cleanly
- [ ] Replace any “mini-pack” wording if the benchmark family becomes broad enough

## Minimal Immediate Conclusion Template

Use this only as a placeholder until the matrix completes.

> The current evidence supports a narrow but real promotion call: on Qwen-family serving lanes, a systems-tuned learned page-format selector can become the default DotCache execution policy without degrading the compact held-out tasks and small held-out QA pack we currently trust. The selector’s runtime overhead is small, the resulting execution path is strongly M3-heavy, and the remaining systems bottleneck lies in backend score/mix cost rather than in selector compute. The cross-family comparison also suggests that selector policy should be family-sensitive rather than globally fixed.
