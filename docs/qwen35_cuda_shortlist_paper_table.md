# Qwen3.5 CUDA Shortlist Paper Table

This note extracts the cleanest committed CUDA shortlist evidence for the Qwen3.5 `0.8B` attention-subset lane and separates the latest rerun from older, more optimistic historical probe notes.

## Scope

- model: `Qwen/Qwen3.5-0.8B`
- backend: `torch_cuda`
- device: local `RTX 5090`
- profile: [`qwen35_0p8b_attention_subset_cuda_third_pass.yaml`](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml)
- decode horizon: `4` generated tokens for the serving table
- shortlist base config:
  - `execution_recent_window=1024`
  - `execution_sink_window=256`
  - `execution_relevance_top_k=4`
  - `execution_relevance_mode=envelope`
- layer-23 context-aware variant:
  - base shortlist plus `layer:23:min_ctx:8192=8`

## Current Rerun To Cite

The current paper-facing artifact is the dedicated rerun in [`qwen35_cuda_shortlist_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl), summarized in the `2026-03-31` entry of [`performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md). This is the latest honest read on the CUDA lane.

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Read |
| ---: | ---: | ---: | ---: | --- |
| `4096` | `257.14` | `164.88` | `163.02` | shortlist win |
| `8192` | `167.94` | `167.37` | `171.77` | effectively flat |
| `16384` | `194.55` | `198.41` | `201.60` | shortlist slightly worse |

What this means:

- the wrapper rerun is valid and reproducible
- the shortlist lane still runs cleanly at all three contexts
- all nine rerun rows stayed on `per_kv_fallback`
- this short-context rerun alone is **not** enough to support a stable long-context win claim
- on the Qwen3.5 CUDA serving path, grouped batching is currently disabled in [`qwen35.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/qwen35.py) before rejection accounting can fire

So the current safe paper claim is narrower:

- shortlist is operational on CUDA
- shortlist is helpful at `4096`
- the `8192/16384` behavior is still unstable across reruns
- the default CUDA serving lane still stays on `per_kv_fallback`
- grouped-batched decode is now operational in the forced ablation lane, but not yet a decisive replacement for the default path
- the large-context `32768/49152` runs need to be presented together with their quality caveat, not as a blanket throughput claim

## Historical Probe Worth Mentioning, Not Leading With

The older March 29 probe note in [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md) is still useful context because it recorded a much stronger earlier result:

| Context | Exact ms/step | Base shortlist ms/step | Speedup |
| ---: | ---: | ---: | ---: |
| `4096` | `416.82` | `205.96` | `2.02x` |
| `8192` | `759.04` | `203.75` | `3.73x` |
| `16384` | `1496.27` | `251.53` | `5.95x` |

That older note is now best treated as an encouraging historical probe, not the current paper table, because the dedicated rerun did not reproduce the `8192` and `16384` wins.

## Quality Spot-Check

The cleanest committed quality spot-check is still the `16384` CUDA read from the older March 29 note.

Base shortlist at `16384`:

- `teacher_forced_logit_max_abs_error=2.5254`
- `teacher_forced_logit_mean_abs_error=0.3031`
- `teacher_forced_logit_rmse=0.3818`
- `replay_output_max_abs_error=0.1198`

Layer-23 context-aware override at `16384`:

- slightly better logit metrics than base shortlist in the first direct probe
- same `replay_output_max_abs_error=0.1198`
- throughput essentially tied in that direct spot-check:
  - base shortlist: `272.60 ms/step`
  - layer-23 override: `271.63 ms/step`

## Large-Context Mixed Result Worth Reporting

There are now committed larger-context artifacts from the clean wrapper path. These are strong enough to discuss in the paper because the serving win is real, but they must be presented as mixed results because the quality read is not yet clean.

Current large-context serving read from the 3-run summary artifact [`qwen35_cuda_shortlist_large_context_repro_serving_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving_summary.md):

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | `n` | Decode-path read |
| ---: | ---: | ---: | ---: | ---: | --- |
| `32768` | `2312.64` one-off exact | `623.88 +/- 13.48` | `626.15 +/- 2.88` | `3` | shortlist rows all `per_kv_fallback` |
| `49152` | `3580.59` one-off exact | `741.45 +/- 26.85` | `792.68 +/- 31.76` | `3` | shortlist rows all `per_kv_fallback` |

Current large-context quality-tail read:

- `32768` still comes from [`qwen35_cuda_shortlist_large_context_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl)
- `49152` now also has a protocol-tagged `held_out` row set in [`qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl)

| Context | Exact tail max abs logit error | Base shortlist | Layer-23 ctx | Read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `0.8984` | `3.5098` | `3.5137` | shortlist much noisier than exact |
| `49152` | `4.5742` | `7.0000` | `6.9648` | protocol-tagged `held_out` synthetic filler run; shortlist still materially worse |

What this adds to the current paper read:

- the shortlist path is a real serving-speed win at `32768` and `49152`
- shortlist page counts remain bounded relative to the full no-shortlist path
- grouped-batched decode still does not activate, and we now know the immediate blocker: the Qwen3.5 CUDA serving integration disables grouped batching up front
- the `49152` quality-tail read is still not clean enough to present as a settled win, even after tagging it under the standardized protocol as `held_out / quality`
- the layer-23 context-aware widening is neutral at `32768` and slightly worse on serving speed at `49152`

## Forced-Grouped CUDA Follow-Up

The natural next question was whether the default CUDA guard was hiding a faster grouped path. The first forced-grouped rerun in [`qwen35_cuda_shortlist_large_context_probe_forced_grouped.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe_forced_grouped.jsonl), using `DOTCACHE_QWEN35_FORCE_GROUPED_BATCHING=1`, was a clear negative result.

| Forced-grouped context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Decode-path read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `2309.26` | `1916.62` | `1843.35` | mixed grouped + fallback |
| `49152` | `3643.76` | `2116.18` | `2059.90` | mixed grouped + fallback |

What this resolves:

- grouped batching is not fundamentally dead on this CUDA lane
- forcing it is dramatically slower than the default path for the shortlist cases
- the first concrete grouped-path blocker is now visible in the counters: `key_value_chunk_signature_mismatch`
- the existing default CUDA guard was directionally right for this workload

After the chunk-schedule split and mixed-signature bucketing fixes, the newer forced-grouped artifact in [`qwen35_cuda_shortlist_large_context_probe_forced_grouped_bucketed.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe_forced_grouped_bucketed.jsonl) materially changes that read:

| Forced-grouped bucketed context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Decode-path read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `2335.11` | `716.36` | `693.11` | all successful rows `grouped_batched=24, per_kv_fallback=0` |
| `49152` | `3658.27` | `777.28` | `766.36` | all rows `grouped_batched=24, per_kv_fallback=0` |

Notes for this follow-up:

- the wrapper matrix had a `NoExactRow` capture miss on `32768 shortlist_base`, so that row comes from the clean direct rerun in [`qwen35_cuda_shortlist_32768_forced_grouped_bucketed_base_single.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32768_forced_grouped_bucketed_base_single.jsonl)
- the previous blocker `key_signature_mismatch_across_groups` disappears on the successful rows
- grouped decode is now fully active in the successful shortlist rows rather than mixed grouped plus fallback
- forced grouped shortlist decode moves much closer to the default non-forced path, but still does not show a decisive win over it

What this changes in the paper read:

- grouped CUDA is now operational on this lane, not merely exploratory
- the default guard is no longer justified by "grouped decode is broken"
- it is still justified by the narrower claim that forced grouped decode has not yet beaten the existing default shortlist path clearly enough to replace it

The two follow-up artifacts after that serving rerun make the story sharper still:

- forced-grouped quality tail from [`qwen35_cuda_shortlist_large_context_quality_tail_forced_grouped.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail_forced_grouped.jsonl)
- 3x serving reproducibility from [`qwen35_cuda_shortlist_large_context_repro_serving/default_repeat1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/default_repeat1.jsonl), [`default_repeat2.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/default_repeat2.jsonl), [`default_repeat3.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/default_repeat3.jsonl), [`forced_grouped_repeat1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/forced_grouped_repeat1.jsonl), [`forced_grouped_repeat2.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/forced_grouped_repeat2.jsonl), and [`forced_grouped_repeat3.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving/forced_grouped_repeat3.jsonl)

Forced-grouped quality-tail summary:

| Context | Base shortlist loss delta / max err | Layer-23 ctx loss delta / max err | Decode-path read |
| ---: | --- | --- | --- |
| `32768` | `-1.33765e-05 / 3.4961` | `-1.42405e-05 / 3.4961` | all rows `grouped_decode_calls=18, per_kv_decode_calls=0` |
| `49152` | `+0.0124781 / 6.9531` | `+0.0121616 / 6.9141` | all rows `grouped_decode_calls=18, per_kv_decode_calls=0` |

What this adds:

- forced grouped decode is now quality-stable enough to evaluate seriously
- it does not repair the existing `49152` loss-tail problem
- quality is broadly comparable to the earlier default non-forced shortlist read, so grouped mode is not the missing quality fix

Serving reproducibility summary:

| Context / case | Default mean ms/step | Forced grouped mean ms/step | Read |
| --- | ---: | ---: | --- |
| `32768 shortlist_base` | `623.88` | `669.76` | grouped slower by `7.35%` |
| `49152 shortlist_base` | `741.45` | `775.01` | grouped slower by `4.53%` |
| `32768 shortlist_l23_ctx` | `626.15` | `672.73` | grouped slower by `7.44%` |
| `49152 shortlist_l23_ctx` | `792.68` | `788.97` | grouped faster by `0.47%` |

Operational caveat:

- both the forced-grouped quality wrapper and the reproducibility wrapper needed one rerun/cleanup pass after partial output files, so the backend story is cleaner than the wrapper story

Bottom line after these follow-ups:

- grouped CUDA is now real, repeatable, and quality-stable
- it still does not justify switching the Qwen3.5 CUDA shortlist lane to grouped-by-default
- the remaining blocker is no longer correctness; it is a narrow performance tradeoff plus the unresolved `49152` quality regime

## Needle Pack

The first reusable named benchmark pack is now in [`qwen35_cuda_needle_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_needle_pack_protocol_v1.jsonl), with the rollup in [`qwen35_cuda_needle_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_needle_pack_protocol_v1_summary.md). This upgrades the earlier single-prompt Needle row into a fixed four-prompt pack with visible variance.

Needle pack summary:

| Context | Case | `n` | Retrieval accuracy | Exact-match rate | Mean decode ms/step | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `32768` | exact | `4` | `1.00` | `0.75` | `2496.10` | `316.83` |
| `32768` | shortlist base | `4` | `1.00` | `1.00` | `561.51` | `144.72` |
| `32768` | shortlist `layer:23` ctx | `4` | `1.00` | `1.00` | `509.53` | `26.05` |
| `49152` | exact | `4` | `1.00` | `0.75` | `3966.83` | `434.54` |
| `49152` | shortlist base | `4` | `1.00` | `0.75` | `759.82` | `176.56` |
| `49152` | shortlist `layer:23` ctx | `4` | `1.00` | `0.75` | `641.14` | `25.20` |

What this adds:

- the branch now has a small fixed named benchmark pack rather than a single successful exemplar
- shortlist keeps a large serving win on a task-style prompt family, not only on exact-length filler
- retrieval stayed correct on all `24/24` rows, including all shortlist rows at `49152`

Important caveats:

- exact-match is not identical to retrieval correctness on this pack, because the `shipment_token` prompt sometimes appends `Question:` after the correct token
- the layer-23 override is now mixed rather than cleanly negative: it is faster across all four prompts at `49152`, but the `32768` result is still prompt-sensitive
- this still does not answer the harder teacher-forced quality-tail issue at `49152`, because Needle is a retrieval-task serving result rather than a loss-tail benchmark

## Passkey Pack

The second reusable named benchmark family is now in [`qwen35_cuda_passkey_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1.jsonl), with the rollup in [`qwen35_cuda_passkey_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1_summary.md). This is a fixed four-prompt RULER-style passkey retrieval pack at the same two contexts.

Passkey pack summary:

| Context | Case | `n` | Retrieval accuracy | Exact-match rate | Mean decode ms/step | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `32768` | exact | `4` | `1.00` | `0.25` | `2390.28` | `200.40` |
| `32768` | shortlist base | `4` | `1.00` | `0.25` | `500.58` | `35.57` |
| `32768` | shortlist `layer:23` ctx | `4` | `1.00` | `0.25` | `511.36` | `40.69` |
| `49152` | exact | `4` | `1.00` | `0.25` | `3823.87` | `386.65` |
| `49152` | shortlist base | `4` | `1.00` | `0.25` | `662.52` | `62.46` |
| `49152` | shortlist `layer:23` ctx | `4` | `1.00` | `0.25` | `657.81` | `40.55` |

What this adds:

- it gives the paper a second named task family rather than a single successful retrieval pack
- retrieval again stayed correct on all `24/24` rows, including all shortlist rows at `49152`
- systems performance is consistent with Needle: shortlist is still about `4.7x` to `5.8x` faster than exact
- shortlist page counts stay stable at `12240` for base and `12336` for `layer:23`

Important caveats:

- strict exact-match is poor even though retrieval is perfect, because the model often emits the correct digits and then continues with prompt text such as `Question: What is` or `Vault record: the`
- this is a RULER-style passkey family, not a full RULER reproduction
- the layer-23 override remains mixed rather than clearly promoted, so it still should not become the headline default shortlist story

## Streaming Window Reference

The first cheap external-style comparator is now in [`qwen35_cuda_streaming_window_needle_pack_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1.jsonl), with the rollup in [`qwen35_cuda_streaming_window_needle_pack_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1_summary.md). This lane uses a simple StreamingLLM-style sink-plus-recent reference policy:

- sink window: `256`
- recent window: `1024`
- no query-aware shortlist expansion

Reference comparison:

| Context | Case | Retrieval accuracy | Exact-match rate | Mean decode ms/step | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: |
| `32768` | exact | `1.00` | `0.75` | `2521.60` | `370.31` |
| `32768` | streaming sink+recent | `0.00` | `0.00` | `156.65` | `18.31` |
| `32768` | shortlist base | `1.00` | `1.00` | `474.23` | `23.75` |
| `49152` | exact | `1.00` | `0.75` | `3909.29` | `440.31` |
| `49152` | streaming sink+recent | `0.00` | `0.00` | `188.55` | `3.21` |
| `49152` | shortlist base | `1.00` | `0.75` | `629.46` | `28.19` |

Why this is useful:

- it gives the paper a real external-style speed/quality frontier instead of only exact-vs-DotCache internal comparisons
- it shows that the cheap window reference is indeed much faster than DotCache shortlist
- it also shows that this speed is unusable on the Needle pack because retrieval collapses completely

Why it still needs caveats:

- this is a StreamingLLM-style reference implementation, not a paper-faithful Quest or H2O reproduction
- the layer-23 override still does not win cleanly enough here to become the promoted default shortlist story

## 49k `top_k=8` Follow-Up

The obvious next ablation was to test whether the `49152` quality problem was simply caused by a shortlist that was too narrow. The clean follow-up artifacts are:

- [`qwen35_cuda_shortlist_49152_topk8_serving_base.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_serving_base.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_quality_base.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_quality_base.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_serving_l23.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_serving_l23.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_quality_l23.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_quality_l23.jsonl)

Compared with the `top_k=4` rerun at `49152`:

| `49152` config | Decode ms/step | Tail loss delta | Tail max abs logit error | Read |
| --- | ---: | ---: | ---: | --- |
| shortlist base, `top_k=4` | `752.13` | `+0.0130062` | `7.0000` | baseline mixed result |
| shortlist base, `top_k=8` | `819.41` serving / `793.73` quality | `+0.0113542` | `6.8711` | modest quality gain, slower serving |
| shortlist `layer:23` ctx, `top_k=8` | `893.25` serving / `1062.99` quality | `+0.0113542` | `6.8711` | no extra quality benefit, clearly slower |

Current interpretation:

- widening from `top_k=4` to `top_k=8` helps the `49152` quality tail a little
- the improvement is not enough to make the run quality-clean
- serving slows down
- the default path still does not activate grouped decode
- once the global shortlist is widened to `top_k=8`, the layer-23 override stops helping and is just extra cost

So `top_k=8` is a useful negative result, not the missing fix.

Why these need caveated presentation:

- the large-context speed story is real, but the quality story is not yet clean at `49152`
- the default path still stays in `per_kv_fallback`, even though the forced-grouped follow-ups now show grouped CUDA can run end-to-end, stay quality-stable, and reach near-parity shortlist throughput
- the paper should present these larger-context rows as the current best systems evidence, but not as a fully locked result yet

## Exact Rerun Path For The CUDA Box

Use the dedicated wrapper script:

```bash
scripts/run_qwen35_cuda_shortlist_paper_table.sh
```

Default output:

```text
benchmarks/results/qwen35_cuda_shortlist_probe.jsonl
```

Custom output:

```bash
scripts/run_qwen35_cuda_shortlist_paper_table.sh \
  benchmarks/results/qwen35_cuda_shortlist_probe_rerun_YYYYMMDD.jsonl
```

The wrapper calls the single-shot runner and regenerates the current `4096/8192/16384` exact-vs-shortlist comparison without the old leaking-process wrapper behavior.

## Raw Source Files

- [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md)
- [`qwen35_cuda_shortlist_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl)
- [`qwen35_cuda_shortlist_large_context_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe.jsonl)
- [`qwen35_cuda_shortlist_large_context_probe_forced_grouped.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe_forced_grouped.jsonl)
- [`qwen35_cuda_shortlist_large_context_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_serving_base.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_serving_base.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_quality_base.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_quality_base.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_serving_l23.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_serving_l23.jsonl)
- [`qwen35_cuda_shortlist_49152_topk8_quality_l23.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_topk8_quality_l23.jsonl)
- [`performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md)
- [`qwen35_cuda_shortlist_ladder_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_ladder_quality_tail.jsonl)
- [`qwen35_cuda_shortlist_32k_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32k_probe.jsonl)
