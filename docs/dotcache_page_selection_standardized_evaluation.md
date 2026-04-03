# DotCache Page Selection Standardized Evaluation Protocol

This note is the working evaluation contract for the page-selection paper and for follow-up research. Its purpose is to stop us from mixing calibration anecdotes, kernel microbenchmarks, and held-out quality results in the same claim bucket.

The broader go or no-go framing for the next stage lives in [dotcache_compressed_page_test_readiness_rfc.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_compressed_page_test_readiness_rfc.md). This protocol is the measurement contract that RFC assumes.

## Goal

Every promoted result should answer four questions unambiguously:

1. Was this row used for tuning, or was it truly held out?
2. Is this a systems claim, a quality claim, or a diagnostic claim?
3. What prompt family produced the row?
4. Which metrics and metadata are required before the row is safe to cite?

It should also answer two newer questions:

5. Is this algorithm truth from a trace or contiguous reference harness, or runtime truth from the real paged serving path?
6. Is the comparison done at a matched effective memory budget, with metadata and fragmentation counted?

## Evaluation Lanes

### 1. Calibration / Discovery

Use for:

- layer-profile search
- rescue-rule search
- shortlist heuristic tuning
- page-routing threshold discovery

Rules:

- never present these rows as the main paper table
- calibration prompts can be synthetic or natural text
- every downstream held-out run must freeze the policy discovered here

### 2. Held-Out Quality

Use for:

- teacher-forced fidelity claims
- loss-tail analysis
- logit drift comparisons
- target-match and token-agreement reporting

Current harness:

- [bench_qwen35_attention_subset_dotcache_loss.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_loss.py)

Required metrics:

- `teacher_forced_loss_delta`
- `teacher_forced_logit_max_abs_error`
- `teacher_forced_token_agreement_rate`
- `teacher_forced_target_match_rate`

Preferred extra metrics:

- `teacher_forced_perplexity_ratio`
- `teacher_forced_logit_mean_abs_error`
- `teacher_forced_logit_rmse`

### 3. Held-Out Systems

Use for:

- decode latency
- throughput
- resident-memory claims
- shortlist load and decode-path comparisons

Current harness:

- [bench_qwen35_attention_subset_dotcache_serving.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py)

Required metrics:

- `dotcache_decode_ms_per_step`
- `ttft_ms` when prompt construction or first-step cost matters
- `p95_decode_ms_per_step`
- `resident_bytes`
- `effective_bytes_per_token`
- shortlist load such as `execution_shortlist_selected_pages` and `execution_shortlist_total_pages`
- decode path counts

Preferred extra metrics:

- `prefill_ms`
- achieved bandwidth
- dense-relative byte ratios
- grouped versus `per_kv_fallback` path counts
- page-format histograms by mode and tensor kind
- repeat statistics: mean, min, max, and confidence interval

### 4. Selector Diagnostics

Use for:

- shortlist recall
- scorer-ranking behavior
- selector overhead decomposition
- grouped-path timing splits

Current harness shapes:

- serving harness with `--recall-analysis`
- serving harness with `--scorer-diagnostic`
- serving harness with `--quality-check`

Diagnostic metrics may explain a result, but they do not replace the held-out quality or systems tables.

Required diagnostic breakdown for promoted selector studies:

- selector recall against exact or oracle references
- selector time
- score time
- shortlist materialization time
- mix or backend-call time

## Prompt Families

### Synthetic Exact-Length Filler

Definition:

- prompts built from a repeated filler unit and trimmed to exact token length

Allowed claims:

- runtime stability
- context-scaling shape
- shortlist page-count behavior
- decode-path activation or fallback behavior

Not sufficient for:

- publication-grade quality claims
- final benchmark tables

### Held-Out Natural Text

Definition:

- fixed prompt slices from natural text that were not used for calibration

Allowed claims:

- teacher-forced quality
- mixed systems/quality tradeoffs

Required for:

- any paper claim that shortlist or routing preserves model quality

### Standardized Long-Context Tasks

Definition:

- named benchmark suites such as LongBench, RULER, or Needle-in-a-Haystack

Allowed claims:

- publication-grade long-context task quality and efficiency

Current status:

- first Needle-in-a-Haystack serving harness is now wired for the Qwen3.5 attention-subset lane
- benchmark-style result tables are still pending; wiring alone does not count as evidence

Required near-term suite shape:

- controlled long-context stress:
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
  - one small multi-instruction and prompt-leakage pack

Needle and passkey remain useful harness checks, but they are not enough for the final default-path decision on their own.

## Model Roles

Model choice should be explicit in every promoted study:

- `Qwen3.5 0.8B` is the wind-tunnel model for selector stress, shortlist scaling, and failure analysis
- at least one stronger Qwen or Llama-family model must carry the final quality-retention verdict

If a study uses only Qwen3.5 0.8B, it should not be framed as the final quality courtroom.

## Required Metadata Per Promoted Row

Every paper-facing row should include:

- `model_id`
- model family label
- backend
- device
- torch dtype
- layer profile or equivalent routing config
- split: `calibration` or `held_out`
- lane: `systems`, `quality`, or `diagnostic`
- prompt family
- dataset or prompt-suite name
- prompt count
- prompt length or context bucket
- batch size
- decode steps or eval steps
- harness truth type: `reference_trace` or `paged_runtime`
- effective-budget accounting rule

Rows that omit prompt count, batch size, or split should be treated as exploratory.

## Required Table Shapes

### Main Systems Table

Use one row per `(model, context, case)`:

| Model | Context | Case | TTFT ms | Decode ms/step | p95 decode ms/step | Effective bytes/token | Selected / candidate pages | Decode path | `n` prompts | Variance |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |

### Main Quality Table

Use one row per `(model, context, case)`:

| Model | Context | Case | Loss delta | Max logit abs error | Token agreement | Target match | `n` prompts | Variance |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |

### Diagnostic Table

Use only when it explains a main result:

| Model | Context | Case | Recall / ranking metric | Selector ms | Score ms | Materialization ms | Mix or backend ms | Interpretation |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |

## Matched-Budget Rule

Comparisons that make memory-efficiency claims must match on effective memory budget rather than payload bytes alone.

Effective memory must count:

- compressed payload bytes
- headers
- scales and zero-points
- exception maps
- exact recent-window pages
- codebooks
- fragmentation and compaction overhead

If a row uses only payload-byte matching, it should be marked exploratory.

## Failure Workbook Rule

For every failed benchmark answer in a promoted selector or format-selection study, keep a compact failure workbook row with exactly these fields:

1. was the exact-attention critical page pruned by read-time selection?
2. was the page kept but damaged by the write-time format?
3. was the page present and healthy but still under-attended downstream?

This workbook is diagnostic, but it is now a required companion for long-context quality debugging.

## Controller Lessons Imported From M0TileDevice

The independent controller work in [/Users/deanocalver/Documents/Projects/M0TileDevice/docs/stage10_stage11_findings_report.md](/Users/deanocalver/Documents/Projects/M0TileDevice/docs/stage10_stage11_findings_report.md) sharpened a few evaluation rules that DotCache should adopt directly.

First, aggregate wins are not enough. A selector or controller family can improve mean quality while still being the wrong default if it fails on worst-case subsets. DotCache promotion decisions should therefore report:

- family-level breakdowns, not just one aggregate mean
- a worst-subset floor or equivalent robustness floor
- whether the candidate is solving the overall allocation problem or merely tuning one friendly family

Second, local repair logic and final controller families should be described differently. The M0TileDevice Stage 10 to Stage 11 pivot is a useful language template:

- a diagnostic branch may uncover real signals and still be the wrong family to promote
- a promoted default should be the first point that clears the robustness gate, not necessarily the highest-gain point on one subset
- a high-gain comparison point can still be worth keeping in the paper as a frontier marker, but it should not be described as the default baseline

Third, selector-oracle studies should expose agreement and allocation structure explicitly. When the relevant data exists, promoted DotCache studies should prefer to include:

- `selection_agreement`
- `bucket_selection_agreement`
- `policy_oracle_gap`
- `allocation_efficiency`
- `policy_only_count`
- `oracle_only_count`
- `top_k_overlap`

These are not all required in every final table, but they are the right diagnostic vocabulary for deciding whether a learned selector is truly matching the oracle or simply finding a different local optimum.

## Promotion Rules

Do not promote a row into the paper's main evidence if any of the following are true:

- it was used during calibration or profile search
- it comes from synthetic-only prompts and is being used for a quality claim
- it lacks prompt count
- it lacks hardware metadata
- it lacks variance or repeat information for a systems claim
- it lacks a clear reference baseline

Additional rule for default-switch claims:

- do not argue that a new execution path should become the default unless it shows a stable advantage across repeated held-out systems runs and does not regress held-out quality
- do not argue that a heterogeneous page-format path should become the default unless it also beats the best fixed single-format DotCache variant on oracle or trace-backed studies
- do not treat a candidate as the new default if it wins on aggregate but still fails a clearly identified worst-subset floor or family-level robustness gate

## Current Status As Of 2026-04-02

What is already in place:

- exact-length serving harnesses
- teacher-forced quality harnesses
- shortlist recall and scorer diagnostics
- grouped versus `per_kv_fallback` path accounting
- step-level timing breakdown fields in the Qwen3.5 serving lane
- a cross-family selector-profile promotion checkpoint in [selector_profile_promotion_checkpoint_20260402.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/selector_profile_promotion_checkpoint_20260402/selector_profile_promotion_checkpoint.md)

What is still missing:

- named benchmark-suite ingestion
- a consistent held-out prompt pack across models
- external baselines reported under the same protocol
- one shared paper-ready table shape across TinyLlama, SmolLM2, Qwen2.5, and Qwen3.5
- an oracle or trace-backed page-format labeling harness
- matched-budget accounting with metadata and fragmentation included
- a failure workbook for benchmark misses
- a standard worst-subset floor and family-breakdown readout for selector-promotion studies
- a standard oracle-gap / agreement diagnostic block for promoted selector rows

The current checkpoint already supports one narrow but real promotion call:

- Qwen3.5 9B can default to the `systems` selector profile for serving
- Llama 3.2 3B currently does not need an extra systems bias because `quality` and `systems` are effectively the same operating point

That means the project is improving on measurement discipline, and the selector-profile question is much less ambiguous than it was one week ago, but it is not yet benchmark-complete.

## Recommended Immediate Use

For the next research cycle:

1. Mark every result in the journal as `calibration`, `held_out`, or `diagnostic`.
2. Stop promoting synthetic-only quality rows into manuscript claims.
3. Require repeat statistics for every new systems comparison.
4. Treat the `49152` quality rescue work as a held-out quality lane, not a calibration anecdote.
5. Add timing-split rows for grouped versus default only as diagnostic support for the main systems table.
6. Treat page-format selection as the main experiment, with trace-backed oracle labels where possible.
