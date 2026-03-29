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
