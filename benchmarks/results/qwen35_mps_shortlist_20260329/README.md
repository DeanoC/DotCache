# Qwen3.5 MPS Shortlist Prototype

## Setup

- Machine: local Mac mini, `torch_mps`
- Model: `Qwen/Qwen3.5-0.8B`
- Lane: exact `M0` only
- Decode config: `tokens_per_page=16`, `max_new_tokens=2`, `profile_backend=true`
- Shortlist config: `execution_recent_window=1024`, `execution_sink_window=256`, `execution_relevance_top_k=4`, `execution_relevance_mode=envelope`

## Artifacts

- [dotcache_exact_m0_envelope_4096.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_4096.jsonl)
- [dotcache_exact_m0_envelope_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_8192.jsonl)
- [dotcache_exact_m0_envelope_quality.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_quality.jsonl)

## Result

The shortlist prototype is active on the real Qwen3.5 serving path and preserves grouped batching on MPS.

At `4096`, decode improved from `687.45 ms/step` on the exact baseline to `608.12 ms/step` with the shortlist, while keeping `decode_path_counts={"grouped_batched": 12, "per_kv_fallback": 0}`. Total attended pages dropped from `6168` candidate pages to `2118` selected pages across the 12 decode invocations.

At `8192`, decode improved from `1320.23 ms/step` on the exact baseline to `520.93 ms/step` with the shortlist, again with fully grouped decoding and no fallback. Total attended pages dropped from `12312` candidate pages to `2104` selected pages across the 12 decode invocations.

The generated token ids matched the exact baseline at both checkpoints:

- `4096`: exact `[65789, 12482]`, shortlist `[65789, 12482]`
- `8192`: exact `[12482, 364]`, shortlist `[12482, 364]`

## Interpretation

This is enough evidence to continue with shortlist work on the real model path. The main constraint held: page reduction can help substantially on this Mac mini without breaking grouped batching. The next iteration should focus on making the scorer less hand-tuned and adding a stronger quality check than token-id equality.

## Quality Check

The serving-quality harness compares the shortlist runtime against the exact dense model using the exact model's own decode inputs as the teacher-forced continuation. This keeps the input sequence aligned while measuring replay and logits drift on the real serving path.

At `4096`, the shortlist kept full top-1 agreement over the two decode steps, with:

- `replay_context_max_abs_error=0.3115`
- `replay_output_max_abs_error=0.1414`
- `teacher_forced_logit_max_abs_error=2.3398`
- `teacher_forced_logit_mean_abs_error=0.3349`
- `teacher_forced_logit_rmse=0.4211`

At `8192`, the shortlist again kept full top-1 agreement, with:

- `replay_context_max_abs_error=0.3253`
- `replay_output_max_abs_error=0.1865`
- `teacher_forced_logit_max_abs_error=2.4434`
- `teacher_forced_logit_mean_abs_error=0.4121`
- `teacher_forced_logit_rmse=0.5184`

The largest replay drift in both runs came from layer `23`, so that is the first place to focus if we want to tighten shortlist quality without giving back the speedup.

## Layer 23 Exact Rerank

I also tested a targeted layer-`23` exact rerank pass that expands the approximate candidate pool for that layer only, then exact-scores the candidate old pages and keeps the same final page budget.

Artifacts:

- [dotcache_exact_m0_envelope_refine_l23.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_refine_l23.jsonl)
- [dotcache_exact_m0_envelope_refine_l23_quality.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_refine_l23_quality.jsonl)

This rerank ran only on layer `23`, with `4` refine invocations total, `32` candidate pages exact-scored, and `16` old pages kept across the two decode steps.

Result:

- `4096` serving got much worse: `608.12 -> 1116.62 ms/step`
- `8192` serving got better: `520.93 -> 455.88 ms/step`
- top-1 agreement stayed perfect at both lengths
- overall quality barely changed

Quality deltas versus the plain shortlist:

- `4096`
  - `teacher_forced_logit_max_abs_error`: unchanged at `2.3398`
  - `teacher_forced_logit_mean_abs_error`: `0.3349 -> 0.3344`
  - `teacher_forced_logit_rmse`: `0.4211 -> 0.4205`
  - `replay_output_max_abs_error`: worsened `0.1414 -> 0.1432`
- `8192`
  - `teacher_forced_logit_max_abs_error`: improved slightly `2.4434 -> 2.4355`
  - `teacher_forced_logit_mean_abs_error`: worsened `0.4121 -> 0.4132`
  - `teacher_forced_logit_rmse`: worsened `0.5184 -> 0.5199`
  - `replay_output_max_abs_error`: worsened `0.1865 -> 0.1904`

Conclusion:

The bounded layer-`23` rerank is not a strong enough quality win to justify enabling it as the default next step. It is worth keeping as an experimental hook, but the better follow-up is probably a smarter approximate scorer or a layer-specific budget policy rather than more exact rescoring.

## Layer 23 Budget Override

I then tried the cheaper version of that idea: keep the approximate shortlist path, but increase the old-page budget for layer `23` only.

Artifacts:

- [dotcache_exact_m0_envelope_l23_budget8.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8.jsonl)
- [dotcache_exact_m0_envelope_l23_budget8_quality.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_quality.jsonl)
- [dotcache_exact_m0_envelope_l23_budget8_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_8192.jsonl)
- [dotcache_exact_m0_envelope_l23_budget8_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_quality_8192.jsonl)

This run used the base shortlist everywhere except layer `23`, where `execution_relevance_top_k` was raised from `4` to `8`.

Result:

- `4096` serving slowed down `608.12 -> 932.07 ms/step`
- `8192` serving improved `520.93 -> 474.50 ms/step`
- grouped batching stayed intact everywhere
- generated token ids still matched the plain shortlist outputs at both lengths

Quality deltas versus the plain shortlist:

- `4096`
  - `teacher_forced_logit_max_abs_error`: improved `2.3398 -> 2.3035`
  - `teacher_forced_logit_mean_abs_error`: improved `0.3349 -> 0.3344`
  - `teacher_forced_logit_rmse`: improved `0.4211 -> 0.4203`
  - `replay_output_max_abs_error`: improved `0.1414 -> 0.1405`
- `8192`
  - `teacher_forced_logit_max_abs_error`: worsened slightly `2.4434 -> 2.4473`
  - `teacher_forced_logit_mean_abs_error`: improved `0.4121 -> 0.4066`
  - `teacher_forced_logit_rmse`: improved `0.5184 -> 0.5118`
  - `replay_output_max_abs_error`: improved `0.1865 -> 0.1826`

The extra budget changed layer `23` from `352` selected pages to `380` selected pages at `8192`, with total selected pages moving from `2104` to `2132`.

Conclusion:

This is the first follow-up that looks better than the plain shortlist on the long-context target. It is still too expensive at `4096`, but at `8192` it improves speed and most quality metrics at the same time. The next sensible step is a context-aware or layer-aware budget policy that only turns the larger layer-`23` budget on once the context is long enough.

## Context-Aware Layer 23 Budget

I then implemented the narrower policy from the previous section: keep the base shortlist everywhere, but only raise layer `23` from `top-k 4` to `top-k 8` once the decode context reaches `8192` tokens.

Artifacts:

- [dotcache_exact_m0_envelope_l23_budget8_minctx8192_4096.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_minctx8192_4096.jsonl)
- [dotcache_exact_m0_envelope_l23_budget8_minctx8192_quality_4096.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_minctx8192_quality_4096.jsonl)
- [dotcache_exact_m0_envelope_l23_budget8_minctx8192_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_minctx8192_8192.jsonl)
- [dotcache_exact_m0_envelope_l23_budget8_minctx8192_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_envelope_l23_budget8_minctx8192_quality_8192.jsonl)

Result:

- `4096`
  - shortlist structure matched the plain shortlist exactly: `2118` selected pages total and `354` selected pages for layer `23`
  - generated token ids still matched the exact baseline: `[65789, 12482]`
  - teacher-forced quality metrics matched the plain shortlist exactly:
    - `teacher_forced_logit_max_abs_error=2.3398`
    - `teacher_forced_logit_mean_abs_error=0.3349`
    - `teacher_forced_logit_rmse=0.4211`
    - `replay_output_max_abs_error=0.1414`
- `8192`
  - shortlist structure matched the always-on layer-`23` budget override: `2132` selected pages total and `380` selected pages for layer `23`
  - generated token ids still matched the exact baseline: `[12482, 364]`
  - quality metrics matched the always-on layer-`23` budget override:
    - `teacher_forced_logit_max_abs_error=2.4473`
    - `teacher_forced_logit_mean_abs_error=0.4066`
    - `teacher_forced_logit_rmse=0.5118`
    - `replay_output_max_abs_error=0.1826`

Measured runtime moved around from run to run, especially at `4096`, but the page-selection statistics confirm that the policy is doing the intended thing: it stays on the cheaper base shortlist at shorter context and only spends the extra layer-`23` budget once the context is long enough to justify it.

Conclusion:

This is the best checkpoint so far. The context-aware policy preserves the `8192` quality/speed tradeoff from the larger layer-`23` budget without changing the shortlist shape at `4096`. The next useful step is to turn this from a benchmark-only knob into a small default heuristic for long-context MPS serving, then sanity-check one higher context if the Mac mini can hold it.

## Default MPS Heuristic

I promoted that context-aware shortlist policy into the Qwen3.5 attention-subset serving path as a default MPS heuristic. It only turns on automatically when:

- backend is `torch_mps`
- the caller has not already set any execution-shortlist knobs
- prompt length is at least `4096`

Artifact:

- [dotcache_exact_m0_default_mps_heuristic.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_default_mps_heuristic.jsonl)

Result:

- short prompts stayed on the exact path:
  - `7` tokens: heuristic disabled, no shortlist pages selected
  - `224` tokens: heuristic disabled, no shortlist pages selected
- exact-length serving prompts picked up the same effective shortlist shape as the tuned benchmark policy:
  - `4096`: heuristic enabled, `1024/256/top-k 4`, layer-`23` context override active, `2118` selected pages, `410.13 ms/step`
  - `8192`: heuristic enabled, same base config plus the layer-`23` `min_ctx:8192=8` expansion, `2132` selected pages, `454.48 ms/step`

Those exact-length results are in the same range as the explicit tuned runs:

- tuned `4096`: `394.58 ms/step`
- defaulted `4096`: `410.13 ms/step`
- tuned `8192`: `466.07 ms/step`
- defaulted `8192`: `454.48 ms/step`

I also tried `16384` with the default heuristic active. It still failed on the Mac mini with MPS OOM during the serving run, so the heuristic improves decode cost but does not move the local memory ceiling enough to make `16384` viable on this machine.

Conclusion:

The default MPS heuristic is now doing the right thing in the real serving path. It stays out of the way for short prompts, automatically picks up the shortlist policy for `4096+`, and matches the tuned long-context behavior closely enough that we can treat it as the new baseline for Mac mini serving experiments. The next higher-context step should happen on a real CUDA host if we want to push past the current `16384` MPS ceiling.

## Shortlist Recall Analysis

I added a serving-side recall-analysis harness that compares the live shortlist against exact per-page attention ranking on the real Qwen3.5 serving path before each decode step.

Artifact:

- [dotcache_exact_m0_default_mps_recall_analysis.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_default_mps_recall_analysis.jsonl)

This run used the default MPS heuristic baseline:

- backend `torch_mps`
- exact `M0` serving path
- `1024` recent tokens
- `256` sink tokens
- base old-page shortlist `top-k 4`
- layer `23` override `min_ctx:8192 -> top-k 8`

Result:

- `4096`
  - `390.11 ms/step`
  - weighted exact-top recall: `0.3663`
  - mean group recall: `0.3658`
  - worst layer by recall: `11`
  - per-layer mean recall:
    - layer `11`: `0.0417`
    - layer `15`: `0.2634`
    - layer `19`: `0.4063`
    - layer `23`: `0.6071`
    - layer `3`: `0.3705`
    - layer `7`: `0.5060`
- `8192`
  - `489.53 ms/step`
  - weighted exact-top recall: `0.3111`
  - mean group recall: `0.2811`
  - worst layer by recall: `11`
  - per-layer mean recall:
    - layer `11`: `0.0357`
    - layer `15`: `0.0625`
    - layer `19`: `0.3571`
    - layer `23`: `0.6227`
    - layer `3`: `0.3348`
    - layer `7`: `0.2738`

The most important finding is that layer `23` is not the worst-recall layer. In fact, it has the best recall at both `4096` and `8192`, even though the earlier replay/logit drift was concentrated there.

Layer `23` detail:

- `4096`
  - step `0`: group recalls `0.5714` and `0.7143`
  - step `1`: group recalls `0.5714` and `0.5714`
  - misses were mostly from the middle-aged pages, with a small old-page tail on the second step
- `8192`
  - step `0`: group recalls `0.6000` and `0.8000`
  - step `1`: group recalls `0.7273` and `0.3636`
  - misses shifted toward more recent old pages rather than only very old pages

Grouped batching is also visibly widening the working set: the union path adds roughly `10-16` pages per layer across the two tested decode steps, including `10-12` extra pages for layer `23`.

Conclusion:

This changes the diagnosis. The approximate scorer is definitely missing many exact top pages overall, but layer `23` is comparatively well served by the shortlist. That means the layer-`23` quality drift is probably not explained by recall failure alone. The next best investigation is a sensitivity/value-side one:

- test whether layer `23` needs more exact value coverage or a different grouped-union policy
- treat layers `11` and `15` as the clearer shortlist-recall failures
- keep CUDA higher-context scaling work separate, since this local result is already enough to redirect the next quality investigation

## Layer Exact Sensitivity Ablation

I added a benchmark-only override that disables shortlist selection for specific layers while leaving the current MPS shortlist baseline active everywhere else.

Artifacts:

- [dotcache_exact_m0_default_mps_exact_l23_quality.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_default_mps_exact_l23_quality.jsonl)
- [dotcache_exact_m0_default_mps_exact_l11_l15_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_default_mps_exact_l11_l15_quality_8192.jsonl)

These runs keep the default MPS shortlist policy in place, but force selected layers back onto full-context exact attention.

### Layer 23 Exact

Compared to the current baseline shortlist:

- `4096`
  - decode cost worsened from `423.37 -> 505.34 ms/step`
  - `teacher_forced_logit_max_abs_error` improved `2.3398 -> 2.2920`
  - `teacher_forced_logit_mean_abs_error` worsened `0.3349 -> 0.3409`
  - `teacher_forced_logit_rmse` worsened `0.4211 -> 0.4286`
  - `replay_output_max_abs_error` improved `0.1414 -> 0.1362`
- `8192`
  - decode cost worsened from `442.87 -> 621.43 ms/step`
  - `teacher_forced_logit_max_abs_error` worsened `2.4473 -> 2.5184`
  - `teacher_forced_logit_mean_abs_error` improved `0.4066 -> 0.3915`
  - `teacher_forced_logit_rmse` improved `0.5118 -> 0.4936`
  - `replay_output_max_abs_error` improved `0.1826 -> 0.1455`

Interpretation:

Making only layer `23` exact is expensive, but it buys back meaningful quality at `8192`, especially on the more stable metrics: mean logit error, RMSE, and replay output error. That is a strong sign that layer `23` is genuinely sensitivity-dominated, not just suffering from poor shortlist recall.

### Layers 11 And 15 Exact

At `8192`, compared to the same shortlist baseline:

- decode cost worsened from `442.87 -> 760.37 ms/step`
- `teacher_forced_logit_max_abs_error` worsened `2.4473 -> 2.4609`
- `teacher_forced_logit_mean_abs_error` improved only slightly `0.4066 -> 0.4044`
- `teacher_forced_logit_rmse` improved only slightly `0.5118 -> 0.5102`
- `replay_output_max_abs_error` worsened `0.1826 -> 0.2080`

Interpretation:

Even though layers `11` and `15` had much worse shortlist recall than layer `23`, making them exact is not a good trade. It costs much more runtime and does not buy enough quality back. That suggests the poor-recall layers are not the main source of the observed downstream drift.

### Conclusion

This ablation sharpens the next direction:

- layer `23` is the quality-sensitive layer worth targeting
- layers `11` and `15` are scorer misses, but not high-value rescue targets
- the next promising experiment is a cheaper layer-`23` fix, not more broad exactness

The most likely useful follow-up is a layer-`23` recency-biased expansion rather than full exact fallback, because the recall probe showed many of the remaining layer-`23` misses at `8192` were in the newer old-page band rather than the deepest history.

## Layer 23 Recent-Window Expansion

I then tried the cheaper variant of the layer-`23` fix: keep shortlist selection active, but enlarge the exact recent window only for layer `23` once the context reaches `8192`.

Artifacts:

- [dotcache_exact_m0_l23_recent1536_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_recent1536_quality_8192.jsonl)
- [dotcache_exact_m0_l23_recent2048_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_recent2048_quality_8192.jsonl)

These runs used the same baseline shortlist policy as before:

- global recent window `1024`
- sink window `256`
- global old-page `top-k 4`
- layer `23` old-page override `min_ctx:8192 -> top-k 8`

The only change was an additional layer-`23` exact recent-window override:

- candidate A: `layer:23:min_ctx:8192=1536`
- candidate B: `layer:23:min_ctx:8192=2048`

Result at `8192` versus the current shortlist baseline (`442.87 ms/step`, layer-`23` selected pages `380`):

- `recent1536`
  - layer-`23` selected pages increased `380 -> 508`
  - total selected pages increased `2132 -> 2260`
  - decode cost worsened `442.87 -> 571.46 ms/step`
  - `teacher_forced_logit_max_abs_error` improved slightly `2.4473 -> 2.4414`
  - `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4095`
  - `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5155`
  - `replay_output_max_abs_error` worsened `0.1826 -> 0.1846`
- `recent2048`
  - layer-`23` selected pages increased `380 -> 634`
  - total selected pages increased `2132 -> 2386`
  - decode cost worsened badly `442.87 -> 1965.74 ms/step`
  - quality metrics were flat-to-worse across the board

Grouped batching stayed intact in both runs, so this is not a fallback-path issue. The expanded recent window really is adding a lot of exact layer-`23` pages, but it does not buy back the quality we want.

Conclusion:

This is not the right cheaper fix. Layer `23` is sensitive, but simply expanding its exact recent band is too expensive and does not materially improve quality. The next better direction is likely a grouped-union/value-side experiment or a more selective layer-`23` page promotion policy rather than a broad recency expansion.

## Layer 23 Per-Group Shortlist Without Grouped Union

I then tested the grouped-union hypothesis directly. This run keeps the shortlist policy, but disables grouped batching for layer `23` only so that layer can decode with its own per-group selected pages rather than the grouped union.

Artifact:

- [dotcache_exact_m0_l23_perkv_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_perkv_quality_8192.jsonl)

Result at `8192` versus the current shortlist baseline:

- layer `23` switched from `grouped_batched` to `per_kv_fallback`
- layer-`23` selected pages dropped from `380` to `356`
- total selected pages dropped from `2132` to `2108`
- decode cost worsened from `442.87 -> 689.02 ms/step`
- `teacher_forced_logit_max_abs_error` worsened `2.4473 -> 2.4668`
- `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4100`
- `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5158`
- `replay_output_max_abs_error` worsened `0.1826 -> 0.1846`

Interpretation:

This is also a negative result. Letting layer `23` use its own per-group shortlist instead of the grouped union does not help quality, even though it reduces the number of pages attended by that layer. It just makes the layer slower.

Conclusion:

The grouped-union constraint does not appear to be the main explanation for the layer-`23` quality drift. At this point the strongest remaining explanation is value sensitivity inside the shortlisted page set itself, or a need for a more selective layer-`23` promotion rule rather than a broad increase in exact pages or a removal of grouped batching.

## Layer 23 Tiny Exact Promotion

I also tried the smallest exact-promotion rule that still changes the shortlist: keep the baseline shortlist policy, but let layer `23` promote just two extra old pages by exact score from a bounded candidate pool while preserving grouped batching.

Artifact:

- [dotcache_exact_m0_l23_exact_promote2_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_exact_promote2_quality_8192.jsonl)

Result at `8192` versus the current shortlist baseline:

- layer-`23` selected pages increased from `380 -> 394`
- total selected pages increased from `2132 -> 2146`
- grouped batching stayed intact for every layer
- decode cost worsened from `442.87 -> 547.83 ms/step`
- `teacher_forced_logit_max_abs_error` worsened `2.4473 -> 2.4590`
- `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4071`
- `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5126`
- `replay_output_max_abs_error` stayed flat at `0.1826`

Conclusion:

This tiny exact promotion is also not the right fix. It is smaller and cheaper than the broad recency expansion or full exact fallback, but it still does not buy back quality. That leaves the main shortlist structure intact and points the next investigation away from “just add a few more exact pages” and toward improving the layer-`23` relevance signal itself.

## Layer 23 Scorer Diagnostic

I added a scorer-diagnostic harness that compares approximate relevance ranking against exact per-page ranking directly, without changing the live shortlist policy.

Artifact:

- [dotcache_exact_m0_default_mps_scorer_diagnostic_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_default_mps_scorer_diagnostic_8192.jsonl)

This run used the current default MPS shortlist baseline at `8192`.

Global result:

- worst layer by approximate-vs-exact top-k recall is still layer `11`
- layer `23` actually has the best scorer alignment:
  - mean rank correlation: `0.9319`
  - mean score-value correlation: `0.9307`
  - mean approx-vs-exact top-k recall: `0.5977`
  - mean exact-top1 approximate rank: `3.0`
  - mean absolute rank error: `33.9`

For comparison, the weaker layers are much worse:

- layer `11`
  - rank correlation: `0.4444`
  - approx-vs-exact top-k recall: `0.0357`
  - exact-top1 approximate rank: `23.5`
  - mean absolute rank error: `103.0`
- layer `15`
  - rank correlation: `0.6465`
  - approx-vs-exact top-k recall: `0.0625`
  - exact-top1 approximate rank: `16.0`
  - mean absolute rank error: `79.1`

Layer `23` detail is more nuanced than the layer averages:

- step `0`, KV group `0`
  - approx-vs-exact top-k recall: `0.6000`
  - exact-top1 approximate rank: `1`
  - first scorer miss at exact rank `4`
  - misses skewed to recent-old pages
- step `0`, KV group `1`
  - approx-vs-exact top-k recall: `0.7000`
  - exact-top1 approximate rank: `4`
  - first scorer miss at exact rank `8`
- step `1`, KV group `0`
  - approx-vs-exact top-k recall: `0.7273`
  - exact-top1 approximate rank: `1`
  - first scorer miss at exact rank `6`
- step `1`, KV group `1`
  - approx-vs-exact top-k recall: `0.3636`
  - exact-top1 approximate rank: `6`
  - approximate top-1 exact rank: `41`
  - misses are concentrated entirely in the recent-old band

Conclusion:

This is the clearest local signal so far. Layer `23` is not globally a bad scorer layer, but one of its KV groups degrades sharply on some steps, and the misses cluster in recent-old pages rather than the deepest history. That points the next iteration toward a conditional layer-`23` scorer improvement, likely on the approximate signal itself, rather than any broader change in shortlist size or batching behavior.

## Layer 23 Recent-Old Scorer Bias

I then tried the simplest scorer-side follow-up: keep the shortlist size fixed, but add a layer-`23` bonus to recent-old pages during approximate relevance ranking.

Artifacts:

- [dotcache_exact_m0_l23_recent_old_bonus05_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_recent_old_bonus05_quality_8192.jsonl)
- [dotcache_exact_m0_l23_recent_old_bonus10_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_recent_old_bonus10_quality_8192.jsonl)

These runs kept the baseline shortlist policy and only added a layer-`23` recent-old score bonus inside the approximate ranker:

- bonus window: `1024` tokens beyond the current recent window
- strengths tested: `0.5` and `1.0`

Result at `8192` versus the current shortlist baseline:

- both scorer-bias runs changed layer `23` from `380` selected pages to `366`
- grouped batching stayed intact
- `bonus=0.5`
  - decode cost worsened `442.87 -> 531.99 ms/step`
  - `teacher_forced_logit_max_abs_error` improved slightly `2.4473 -> 2.4434`
  - `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4121`
  - `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5185`
  - `replay_output_max_abs_error` worsened `0.1826 -> 0.1885`
- `bonus=1.0`
  - decode cost exploded `442.87 -> 2798.49 ms/step`
  - quality metrics were again flat-to-worse on the stable measures

Interpretation:

The underlying idea is directionally plausible, but this implementation is not the answer. A plain recent-old bonus is too blunt, and in this prototype it is also too expensive because it falls back to more Python-side per-page scoring work. It does not buy back the quality we want on layer `23`.

Conclusion:

The next scorer iteration needs to be more conditional than a fixed recency bias. The evidence now points toward a layer-`23` KV-group-specific or confidence-triggered scorer adjustment, rather than any blanket increase in exact pages, recent window, or recent-old bonus.

## Layer 23 Confidence-Gated Exact Promotion

I then tried a confidence-gated version of the earlier exact-promotion experiment: promote two extra exact pages for layer `23`, but only when the approximate shortlist boundary margin looks uncertain.

Artifact:

- [dotcache_exact_m0_l23_confidence_promote2_margin025_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_confidence_promote2_margin025_quality_8192.jsonl)

This run used:

- `execution_exact_promote_top_k=2`
- `execution_exact_promote_margin_threshold=0.25`
- layer `23` only

Result at `8192` versus the current shortlist baseline:

- layer-`23` selected pages increased from `380 -> 394`
- grouped batching stayed intact
- decode cost worsened `442.87 -> 533.57 ms/step`
- `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4071`
- `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5126`
- `replay_output_max_abs_error` stayed flat at `0.1826`

This effectively matched the always-on `promote2` experiment, so I reran the scorer diagnostic with the new boundary-margin metric:

- [dotcache_exact_m0_default_mps_scorer_diagnostic_8192_v2.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_default_mps_scorer_diagnostic_8192_v2.jsonl)

For layer `23`, the normalized approximate boundary margins were:

- step `0`, group `0`: `0.0134`
- step `0`, group `1`: `0.0558`
- step `1`, group `0`: `0.0215`
- step `1`, group `1`: `0.0847`

The important part is that the worst group from the earlier scorer analysis is actually the highest-margin one, not the lowest-margin one:

- step `1`, group `1`
  - approx-vs-exact top-k recall: `0.3636`
  - approximate top-1 exact rank: `41`
  - normalized boundary margin: `0.0847`

Conclusion:

Simple confidence gating on shortlist boundary margin is not the right detector. The bad layer-`23` group can still look fairly confident while being wrong. That means the next useful experiment is probably a group-specific secondary feature or a score-disagreement trigger, not a generic low-margin rescue rule.

## Layer 23 Dual-Scorer Rescue

I then tried a true second-feature rescue for layer `23`: keep envelope scoring as the main shortlist signal, compute a sketch ranking alongside it, and only add a couple of sketch-selected pages when the two top-k sets disagree enough.

Artifacts:

- [dotcache_exact_m0_l23_dualscore_sketch2_overlap05_diag_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_dualscore_sketch2_overlap05_diag_8192.jsonl)
- [dotcache_exact_m0_l23_dualscore_sketch2_overlap05_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_dualscore_sketch2_overlap05_quality_8192.jsonl)

This run used:

- primary relevance mode: `envelope`
- secondary relevance mode: `sketch`
- trigger: layer-`23` only
- trigger condition: primary/secondary top-k overlap below `0.5`
- rescue size: add up to `2` secondary-selected pages

The diagnostic was encouraging in one narrow sense: it found the bad layer-`23` case we care about.

- layer `23` secondary trigger rate: `0.25`
- the trigger fired on step `1`, KV group `1`
- for that group, primary approx-vs-exact recall was `0.3077`
- the secondary sketch ranker improved exact-top recall for that same group to `0.3846`
- approximate top-1 exact rank for the primary scorer was `41`, while the secondary scorer's top-1 exact rank was `2`

But the actual serving tradeoff was still bad at `8192` versus the current shortlist baseline:

- grouped batching stayed intact
- layer-`23` selected pages only increased `380 -> 384`
- total selected pages only increased `2132 -> 2136`
- decode cost worsened `442.87 -> 555.72 ms/step`
- `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4074`
- `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5128`
- `teacher_forced_logit_max_abs_error` improved only trivially `2.4473 -> 2.4453`
- `replay_output_max_abs_error` stayed flat at `0.1826`

Conclusion:

The disagreement trigger is directionally better than the margin trigger because it really does identify the bad layer-`23` group. But this prototype is still not a serving win. The extra Python-side secondary scoring cost overwhelms the tiny page-set change, and the quality lift is too small to matter. That shifts the next question from "can a second feature detect the problem?" to "can we make a second feature cheap enough to use, or distill it into a cheaper layer-23 signal?"

## Layer 23 Cheap Neighbor Rescue

I then tried a cheaper distilled rescue that uses only the primary envelope ranking. The idea was: when layer `23`'s top-k looks split between a cluster of very early anchor pages and a recent-old band, add a couple of adjacent predecessor pages just before the selected recent-old cluster.

Artifacts:

- [dotcache_exact_m0_l23_neighbor2_anchor1024_min4_diag_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_neighbor2_anchor1024_min4_diag_8192.jsonl)
- [dotcache_exact_m0_l23_neighbor2_anchor1024_min4_quality_8192.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/dotcache_exact_m0_l23_neighbor2_anchor1024_min4_quality_8192.jsonl)

This run used:

- layer `23` only
- trigger when the primary top-k contains at least `4` anchor pages with `token_end <= 1024`
- require at least one recent-old page in the primary top-k
- add up to `2` predecessor pages immediately before the selected recent-old block

The diagnostic shows the trigger is cheap and targeted:

- layer-`23` trigger rate: `0.5`
- it fired on step `0`, group `1`, and step `1`, group `1`
- it did not fire on the healthier layer-`23` groups
- unlike the dual-scorer prototype, this rescue does not need a second relevance pass

But the quality run at `8192` was still slightly worse than the current shortlist baseline:

- grouped batching stayed intact
- layer-`23` selected pages increased `380 -> 384`
- total selected pages increased `2132 -> 2136`
- decode cost worsened slightly `442.87 -> 451.67 ms/step`
- `teacher_forced_logit_mean_abs_error` worsened `0.4066 -> 0.4071`
- `teacher_forced_logit_rmse` worsened `0.5118 -> 0.5124`
- `teacher_forced_logit_max_abs_error` worsened `2.4473 -> 2.4492`
- `replay_output_max_abs_error` worsened `0.1826 -> 0.1846`

Conclusion:

This is a better direction than the live dual-scorer rescue because it captures the intended pattern without the big secondary-scoring overhead. But even this cheaper version still does not buy back enough quality to justify changing the current shortlist policy. The next useful step is probably offline distillation: use the exact-vs-approx diagnostics to learn a cheaper layer-`23` signal from the existing envelope features, rather than hand-authoring more page-addition rules.
