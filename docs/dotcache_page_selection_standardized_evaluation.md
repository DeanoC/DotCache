# DotCache Page Selection Standardized Evaluation Protocol

This note is the working evaluation contract for the page-selection paper and for follow-up research. Its purpose is to stop us from mixing calibration anecdotes, kernel microbenchmarks, and held-out quality results in the same claim bucket.

## Goal

Every promoted result should answer four questions unambiguously:

1. Was this row used for tuning, or was it truly held out?
2. Is this a systems claim, a quality claim, or a diagnostic claim?
3. What prompt family produced the row?
4. Which metrics and metadata are required before the row is safe to cite?

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
- `resident_bytes`
- shortlist load such as `execution_shortlist_selected_pages` and `execution_shortlist_total_pages`
- decode path counts

Preferred extra metrics:

- `prefill_ms`
- dense-relative byte ratios
- grouped versus `per_kv_fallback` path counts
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

Rows that omit prompt count, batch size, or split should be treated as exploratory.

## Required Table Shapes

### Main Systems Table

Use one row per `(model, context, case)`:

| Model | Context | Case | Decode ms/step | Bytes vs dense | Selected / candidate pages | Decode path | `n` prompts | Variance |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |

### Main Quality Table

Use one row per `(model, context, case)`:

| Model | Context | Case | Loss delta | Max logit abs error | Token agreement | Target match | `n` prompts | Variance |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |

### Diagnostic Table

Use only when it explains a main result:

| Model | Context | Case | Recall / ranking metric | Selection ms | Materialization ms | Backend-call non-backend ms | Interpretation |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |

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

## Current Status As Of 2026-03-31

What is already in place:

- exact-length serving harnesses
- teacher-forced quality harnesses
- shortlist recall and scorer diagnostics
- grouped versus `per_kv_fallback` path accounting
- step-level timing breakdown fields in the Qwen3.5 serving lane

What is still missing:

- named benchmark-suite ingestion
- a consistent held-out prompt pack across models
- external baselines reported under the same protocol
- one shared paper-ready table shape across TinyLlama, SmolLM2, Qwen2.5, and Qwen3.5

That means the project is improving on measurement discipline, but it is not yet benchmark-complete.

## Recommended Immediate Use

For the next research cycle:

1. Mark every result in the journal as `calibration`, `held_out`, or `diagnostic`.
2. Stop promoting synthetic-only quality rows into manuscript claims.
3. Require repeat statistics for every new systems comparison.
4. Treat the `49152` quality rescue work as a held-out quality lane, not a calibration anecdote.
5. Add timing-split rows for grouped versus default only as diagnostic support for the main systems table.
