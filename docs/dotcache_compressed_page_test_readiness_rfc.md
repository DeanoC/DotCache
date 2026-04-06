# DotCache Compressed-Page Test-Readiness RFC

This RFC defines the next-stage decision that DotCache needs to support. It is intentionally narrower than "keep improving heuristics." The question is whether a small heterogeneous menu of compressed page formats can become a default long-context serving path, or whether DotCache is still only a promising local optimization.

It builds on the current manuscript in [dotcache_layer_page_selection.md](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md), the standardized contract in [dotcache_page_selection_standardized_evaluation.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_page_selection_standardized_evaluation.md), and the execution roadmap in [dotcache_submission_execution_plan.md](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_submission_execution_plan.md).

## 1. Decision To Support

The next stage should answer one concrete question:

- does learned or calibrated page-format selection beat fixed-format DotCache and strong external baselines at matched effective memory budgets, without leaning on weak benchmark slices?

The current evidence says DotCache is worth taking seriously, but not yet properly evidenced:

- compressed-domain KV execution is real, not just a storage trick
- Qwen3.5 shortlist serving wins at `32768` and `49152` are material
- Needle and passkey retrieval can remain clean
- LongBench-style QA retention is still not good enough
- the `49152` quality tail remains unresolved

That means the scientific object is no longer "another routing tweak." It is page-format selection itself.

## 2. Freeze The Evaluation Contract Before Changing The Model

The official evaluation contract should remain four-lane:

1. `calibration / discovery`
2. `held-out quality`
3. `held-out systems`
4. `selector diagnostics`

Every promoted row should now also report:

- `ttft_ms`
- `p95_decode_ms_per_step`
- `effective_bytes_per_token`
- page-format histograms by mode and tensor kind
- selector recall against an exact or oracle reference
- separate totals for selector time, score time, and mix time

The contract must also preserve a hard split between:

- algorithm truth from a contiguous or trace-driven reference harness
- runtime truth from the real paged runtime

This split matters because selector cost can look tiny when returns are view-like, yet naive materialization can dominate the same path in the real loop.

## 3. Model Roles

Qwen3.5 0.8B should stay in the roster, but with a narrower role:

- use it as the wind tunnel for shortlist scaling, selector stress, and failure analysis
- do not use it as the final courtroom for quality retention

Why:

- only six layers use full attention, so shortlist effects are unusually large and easy to observe
- its current LongBench QA baseline is already too weak to carry the final quality verdict

The final quality verdict should therefore include at least one stronger Qwen or Llama-family model.

## 4. Make Page-Format Determination The Main Experiment

The current bundle names are implementation shorthands, not the core experiment. The main experiment should be an offline oracle for cheapest-safe page formatting.

### Proposed oracle workflow

1. capture full-precision traces for a small fixed model roster
2. replay each page under candidate modes:
   - `M0-2b`
   - `M0-3b`
   - `M0-4b`
   - `M1-4b`
   - `M2-4b`
   - `M4-4b`
   - `M3 exact`
   - `T3` when stable
3. label each page with the cheapest safe format under frozen fidelity thresholds
4. compare three selectors against that oracle:
   - current heuristics
   - a static layer/head/age map
   - a lightweight learned predictor

### Page-local targets

The offline target should combine cheap local measures with one downstream answer check:

- attention-logit delta
- top-k attention agreement
- next-token logprob delta
- downstream answer delta on held-out prompts

The menu already exists in the repo. The missing part is treating format selection as the main supervised or calibrated object.

## 5. Baselines

Baselines should arrive in two waves so the project does not stall on perfect coverage.

### Wave 1: canonical compression and serving comparators

Compression-side comparators:

- `KIVI`
- `KVQuant`
- `QJL`
- `QServe`
- `InnerQ`
- `RotateKV`
- `PM-KVQ`
- one saliency-aware mixed-precision representative such as `ZipCache` or `MiKV`

Read-time or shortlist comparators:

- `Quest`
- `H2O`
- `SnapKV`
- `PQCache`
- StreamingLLM sink-plus-recent window

Stretch additions if wiring cost is reasonable:

- `Expected Attention`
- `ParisKV`
- `Self-Indexing KVCache`

### Wave 2: methods closest to DotCache's actual design pressure

- `TailorKV`
- `Kitty`
- `DiffKV`

These second-wave baselines matter because page-local precision, irregular memory layout, and compaction are not incidental runtime details. They are part of the algorithmic claim.

## 6. Benchmark Suite

The benchmark suite should stay small, sharp, and hard to game.

### Controlled long-context stress

- `RULER`
- `NoLiMa`

### Realistic long-context understanding

- `LongBench v2`
- `LongBench Pro`
- `InfinityBench`

### Reporting discipline references

- `HELMET`
- `100-LongBench`

These can shape reporting discipline even if the full suites are not run immediately.

### Decode-heavy reasoning slice

At minimum:

- `GSM8K`
- `MATH500`

Compression behavior on long reasoning traces can differ materially from prefill-heavy retrieval tasks, so this slice should be mandatory rather than optional.

### Deployment gate slice

Add one small pack for:

- multi-instruction following
- system-prompt leakage

This is a deployment gate, not a leaderboard play. KV compression can selectively erase instructions or amplify leakage while leaving average scores deceptively calm.

## 7. Target The Failure Mode Directly

The current Qwen3.5 HotpotQA-style diagnostic already points to repeated old-page misses. The next stage should produce a page-level failure workbook for every benchmark miss.

For each failed answer, log only:

1. whether the exact-attention critical page was pruned by read-time selection
2. whether it was kept but damaged by the write-time format
3. whether it was present and healthy, but still under-attended downstream

This turns "quality regressed" into a diagnosis tree:

- selector scoring problem
- format-selection problem
- downstream budget-allocation or attention-use problem

## 8. Separate Algorithm Truth From Runtime Truth

Every promising selector should be tested twice:

1. in a trace-driven contiguous reference harness
2. in the real paged runtime

Required runtime reporting:

- `ttft_ms`
- mean decode latency
- `p95` decode latency
- batch scaling
- achieved bandwidth
- `effective_bytes_per_token`

Effective memory must count more than compressed payload bytes. It should include:

- headers
- scales and zero-points
- exception maps
- exact recent-window pages
- codebooks
- fragmentation and compaction overhead

This is especially important once differentiated precision or mixed page modes introduce layout irregularity.

## 9. Confidence Gates

DotCache should not be called properly evidenced until it clears all four gates:

### Gate A: format-menu truth

- a heterogeneous page-format menu beats the best single fixed format by a material margin on oracle traces

### Gate B: selector truth

- the learned or calibrated selector stays close to oracle and repairs the current benchmark pain rather than only improving synthetic retrieval

### Gate C: matched-budget truth

- at matched effective memory budgets, DotCache beats at least the canonical external set on one strong quality model and one systems model

### Gate D: deployment truth

- no severe `TTFT` cliff
- no `p95` disaster
- no obvious instruction-following or prompt-leakage regression

Concrete internal bar:

- no named slice worse than `5` absolute points versus exact on the same model
- small aggregate quality loss
- and either a clear `>2x` long-context decode win or a better same-quality memory/throughput tradeoff after metadata is counted

## 10. Immediate Implementation Order

This should be the actual execution order for the next stage:

1. freeze the evaluation contract and logging schema
2. build the full-precision trace recorder and page-oracle replay harness
3. run the current DotCache page menu across page-size and mode sweeps
4. add wave-1 baselines through adapters or shared libraries where possible
5. run the benchmark suite and generate the page-level failure workbook
6. promote only the selectors that survive the matched-budget bakeoff into the real serving path

## 11. Current Checkpoint

The current repo checkpoint is summarized in:

- [selector_profile_promotion_checkpoint_20260402.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/selector_profile_promotion_checkpoint_20260402/selector_profile_promotion_checkpoint.md)

The present promotion call is:

- Qwen3.5 9B: `systems` is the correct default serving profile
- Llama 3.2 3B: `quality` and `systems` are effectively the same operating point today

Why this is enough for a local promotion call:

- task-level success is preserved across instruction, retrieval, and reasoning slices
- Qwen keeps the large serving-speed win under the systems profile
- Llama confirms that the same selector machinery does not require a forced systems bias when the learned selector is already saturated to `M3`

What is still not claimed:

- that DotCache has already cleared the full matched-budget external-baseline courtroom
- that every model family should inherit the same systems bias by default

## 12. Bottom Line

The pivot is straightforward:

- stop treating page selection as a support component
- make page-format selection the experiment

The current DotCache artifacts already show enough systems signal to justify that pivot. They do not yet show enough held-out evidence to claim benchmark-grade confidence. This RFC is the bridge from interesting machinery to a test-ready research program.
