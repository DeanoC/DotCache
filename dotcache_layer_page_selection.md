# Layer-Aware Page Selection in DotCache

_Working manuscript source reconstructed from `dotcache_layer_page_selection.pdf` on 2026-03-31._

## Abstract

DotCache proposes executing attention directly on page-organized, low-bit key-value (KV) caches rather than dequantizing compressed tokens back to full precision before the attention kernel runs. A critical but underexplored part of that design is the selection subsystem that decides which codec, bitwidth, and execution mode to assign to each page. This note describes the current DotCache selection stack: write-time page routing, recent-window escape, and read-time shortlist gating. The current artifact supports six page modes (`M0`, `M1`, `M2`, `M3`, `M4`, `T3`) and has been exercised on TinyLlama, SmolLM2, Qwen2.5, and Qwen3.5. The strongest current systems result is that query-aware shortlisting can produce large serving-speed wins at long context on the Qwen3.5 CUDA lane, including clean serving wins at `32768` and `49152` tokens. The strongest current caveat is that the same lane still shows unstable long-context quality, and simple rescue heuristics such as widening `top_k` or adding a layer-23 override do not make the `49152` regime quality-clean. The main engineering conclusion is therefore narrower than the original draft implied: layer-aware selection is necessary and promising, but the present evidence is still prototype-grade and not yet sufficient for a publication-ready benchmark claim.

## 1. Introduction

DotCache treats low-bit KV representations as an execution format rather than a storage format. Instead of compressing keys and values for memory savings and then widening them back to full precision before attention, DotCache tries to keep decode fused with score and mix computation. That design targets the widening and metadata traffic that recent serving work has identified as a dominant overhead in low-bit pipelines.

Compressed-domain execution creates a systems question that storage-only quantization papers can mostly sidestep: which codec should each page use, and when should the runtime bypass compression entirely? In DotCache, the answer propagates into the attention kernel itself because the kernel consumes compressed pages directly. A bad per-layer or per-page choice is not normalized away by a generic dequantization stage later.

This draft focuses on that selection layer: the write-time policy that maps observed page statistics to a page mode, the recent-window escape that keeps the hottest tokens in high precision, and the read-time shortlist that limits which old pages are attended. The core claim is modest and specific. DotCache is interesting because it combines compressed-domain execution with adaptive page routing. The current artifact does not yet prove that this combination beats the best long-context baselines on standard benchmark suites, but it does show a meaningful systems pattern: once context is large enough, page selection can dominate the runtime story, while quality recovery becomes the new limiting factor.

## 2. The Page Selection Problem

A DotCache page is the atomic unit of compressed KV storage. Each page is indexed by `(layer, head, token_range, kind)`, where `kind` distinguishes key pages from value pages. Within a page, vectors are partitioned into fixed-width groups along the inner dimension, usually 32 channels per group. Each page is then encoded with one of six modes:

| Mode | Codec family | Current role | Tensor family |
| --- | --- | --- | --- |
| `M0` | Affine quantization (`scale + bias`) | Default workhorse at 2-4 bits | K and V |
| `M1` | Nonuniform LUT quantization | Cluster-friendly value path | Primarily V |
| `M2` | Sketch-plus-residual | Inner-product-aware key scoring | K only |
| `M3` | High-precision escape | Recent tokens and explicit rescue pages | K and V |
| `M4` | Projected-basis key encoding | Structured key compression | K only |
| `T3` | TurboQuant-style vector quantization | Experimental aggressive 3-bit path | K and V |

The selection problem is: given a page of newly produced key or value vectors, choose the mode and bitwidth that preserve useful attention behavior while staying within a memory and runtime budget. This is not a global quantization decision. The choice depends on the producing layer, tensor kind, token age, and the measured statistics of the page itself.

## 3. Write-Time Selection

### 3.1. Policy Bundles, Not a Monotonic Ladder

The March 2026 artifact exposed a terminology problem in the original PDF draft. It described `exact`, `strict`, `balanced`, and `aggressive` as a clean sensitivity ladder. That is not correct in the current implementation.

In code, these labels are named policy bundles with tensor-kind-specific candidate orderings. They are only partially ordered. In particular, `strict` is not uniformly "more conservative" than `balanced` for value pages because the current `strict` value bundle tries `M1/4b` before `M0/4b`, while `balanced` begins with `M0/3b` and may avoid `M1` entirely. That planner interaction is exactly what hurt SmolLM2 in the original artifact.

The current bundle definitions in [`planner.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/planner.py) are:

| Bundle label | Key candidate order | Value candidate order | Notes |
| --- | --- | --- | --- |
| `exact` | default mode only | default mode only | Disables adaptive downgrade; this is a policy label, not the same thing as `M3` |
| `strict` | `M0/affine/4` | `M1/lut/4 -> M0/affine/4` | Conservative on K, but not globally most conservative on V |
| `balanced` | `M0/affine/2 -> M2(or M4)/4 -> M0/affine/4` | `M0/affine/3 -> M1/lut/4 -> M0/affine/4` | Default adaptive policy |
| `aggressive` | `M0/affine/2 -> M2(or M4)/4 -> M0/affine/4` with relaxed thresholds | `M0/affine/2 -> M0/affine/3 -> M1/lut/4 -> M0/affine/4` with relaxed thresholds | Wider approximation envelope |

That wording fixes two reviewer-visible issues at once:

1. `exact` now means "no adaptive mode search inside this policy bundle."
2. `M3 escape` is reserved for explicit high-precision storage, recent-window pages, or hand-pinned rescue pages.

### 3.2. Observed Statistics and Routing Rule

Write-time selection uses a single stats pass over each page. The current implementation computes:

- `rms`
- `abs_max`
- `outlier_fraction`
- `channel_range_mean`

The stats extractor is `observe_page(...)` in [`planner.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/planner.py). The current routing rule can be summarized as:

```text
observe_page(page) -> {rms, abs_max, outlier_fraction, channel_range_mean}

if token_age < recent_window:
    return M3 escape

for candidate in policy_bundle(kind, bundle_label):
    if candidate is allowed under the page statistics:
        return candidate

return safest candidate in the bundle
```

The current allow/deny logic is heuristic rather than learned:

- `M2` and `M4` are allowed only on key pages and depend on `outlier_fraction` and `channel_range_mean`.
- `M1` is allowed only on value pages and depends on the same thresholds.
- `M0/2b` depends mostly on `outlier_fraction` and `abs_max`.
- `M0/3b` uses a slightly relaxed version of those thresholds.
- `M0/4b` is always permitted as the safe fallback.

This is exactly where the paper should be honest. The mechanism is clear and reproducible from code, but the threshold values are still hand-authored. A publication draft needs either automated search or a calibration/test split for profile discovery.

### 3.3. Recent-Window Escape

The recent window is a separate mechanism from the policy bundle labels. Pages younger than `recent_window` tokens are routed to `M3` with an explicit escape dtype, usually `float16`. The current code path does this before bundle evaluation.

That distinction matters for the manuscript:

- `exact` is a routing policy label.
- `M3 escape` is a storage and execution mode.
- "exact baseline" should be reserved for the no-shortlist, no-approximation runtime configuration.

## 4. Read-Time Selection

### 4.1. Windowed Shortlisting

DotCache also performs read-time selection during decode. The selector combines:

- a sink window for early positional anchors
- a recent window for the newest pages
- relevance gating over the old middle region

For relevance gating, the current artifact supports two lightweight page summaries:

- sketch scoring, where each page stores a compact mean vector or sketch row
- envelope scoring, where each page stores per-channel minima and maxima and computes a query-conditioned upper bound on the page score

The current auto profile in [`execution_profiles.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/execution_profiles.py) uses:

- `sink_window_tokens=256`
- `recent_window_tokens=1024` at `<=4096` context
- `recent_window_tokens=2048` at `<=8192` context
- `recent_window_tokens=4096` beyond `8192`
- `relevance_top_k=4` at shorter contexts and `8` at longer ones

### 4.2. Complexity Claim, Restated Precisely

The original PDF said that shortlisting yields sub-linear attention cost. That statement needs tighter wording.

What is currently true:

- the **full attention work over selected pages** can grow sub-linearly in total context if the shortlist remains bounded or grows slowly
- the **current software selector pass** still scores every candidate old page, so the selector itself is linear in candidate-page count
- total end-to-end decode cost is therefore only sub-linear if selector overhead stays small compared with the saved attention work

That should be the paper's actual claim. The artifact currently supports "sub-linear selected-set attention" rather than "globally sub-linear decode."

### 4.3. Current Selector Overhead Evidence

The repo now contains an explicit selector replay microbenchmark in [`qwen35_selector_replay_microbench_smoke.json`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_selector_replay_microbench_smoke.json). On a `49,152`-token Qwen3.5 replay trace with `2,992` candidate pages and `81` selected pages per call, the measured Python/NumPy selector costs were:

| Selector path | Mean total ms/call | Mean compute ms/call | Extra materialization ms/call |
| --- | ---: | ---: | ---: |
| `score_all_pages` | `0.0182` | `0.0182` | `0.0000` |
| `candidate_view` | `0.0164` | `0.0160` | `0.0004` |
| `candidate_packedspan` | `0.0166` | `0.0162` | `0.0004` |
| `candidate_take` | `0.0690` | `0.0179` | `0.0511` |

Two points matter:

1. The present selector pass is cheap enough to matter, but it is not free.
2. Copying the selected span naively can dominate selector cost; view-like or packed-span returns are much better.

This is useful evidence, but it is still not the end-to-end overhead study a paper needs. We still need selector-plus-attention decomposition inside the actual serving loop.

## 5. Current Empirical Evidence

### 5.1. Standardized Evaluation Protocol

The repo now needs to treat evaluation as a protocol rather than a pile of ad hoc probes. The right standardized contract has four lanes:

1. `calibration / discovery`
   Used only to tune bundle thresholds, shortlist heuristics, or rescue rules. These runs are not main-table evidence.
2. `held-out quality`
   Teacher-forced exact-length runs on held-out prompts. This lane supports quality claims.
3. `held-out systems`
   Serving-style exact-length runs on the same held-out prompt family. This lane supports throughput and memory claims.
4. `selector diagnostics`
   Optional recall and timing instrumentation used to explain why a systems result happened. This lane is diagnostic, not a substitute for quality or throughput tables.

The repo-side version of this contract now lives in [`dotcache_page_selection_standardized_evaluation.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_page_selection_standardized_evaluation.md) so future experiments can use the same reporting rules as the paper draft.

For DotCache page selection, every reported experiment should therefore declare:

- split: `calibration` or `held_out`
- lane: `quality`, `systems`, or `diagnostic`
- prompt source: synthetic exact-length filler, held-out natural text, or standardized long-context task
- model / backend / device / dtype
- batch size
- decode horizon or teacher-forced evaluation horizon
- number of prompts
- whether variance or confidence intervals are reported

The current harnesses already map cleanly onto that structure:

| Lane | Current harness shape | Current purpose | Publication status |
| --- | --- | --- | --- |
| `systems` | [`bench_qwen35_attention_subset_dotcache_serving.py`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py) | exact-length serving throughput, resident bytes, shortlist counts, decode path | good for controlled systems probes; not sufficient alone for publication |
| `quality` | [`bench_qwen35_attention_subset_dotcache_loss.py`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_loss.py) | teacher-forced loss, logit drift, agreement on fixed slices | good for held-out quality checks; named benchmark quality ingestion is still missing |
| `diagnostic` | serving harness with `--quality-check`, `--recall-analysis`, or `--scorer-diagnostic` | replay drift, shortlist recall, scorer ranking, timing breakdowns | diagnostic support only |
| `calibration` | local profile sweeps and layer-specific probe scripts | threshold and override discovery | explicitly non-mainline evidence |

The prompt source itself must also be classified. Going forward, DotCache should use three prompt families with different evidentiary weight:

| Prompt family | Allowed claim | Current status |
| --- | --- | --- |
| synthetic exact-length filler | kernel/runtime stability, shortlist scaling, decode-path behavior | implemented today |
| held-out natural text | quality and mixed systems-quality claims | partially implemented through exact-length teacher-forced harnesses, but not yet packaged as a named suite |
| standardized long-context tasks | publication-grade long-context task claims | two fixed four-prompt synthetic retrieval packs now exist (`Needle` and RULER-style `passkey`), and one small non-synthetic LongBench-derived QA mini-pack now exists, but broader suite coverage and quality-clean benchmark rows are still missing |

This matters because several current harnesses still construct prompts by repeating a synthetic filler string (`"Cache locality matters for fast decoding."`) and trimming to exact token length. That is acceptable for systems microbenchmarks, but not for the headline quality claim of a paper.

### 5.2. Metric Families and Reporting Schema

The paper should stop mixing incomparable measurements in one summary table. Each run should report one of three metric families.

`systems` metrics:

- `dotcache_decode_ms_per_step`
- `prefill_ms` when relevant
- `resident_bytes`, `kv_resident_bytes`, and any dense-relative byte ratio
- shortlist load such as `execution_shortlist_selected_pages` and `execution_shortlist_total_pages`
- decode path counts such as grouped vs `per_kv_fallback`
- hardware metadata, dtype, batch size, prompt count, and variance

`quality` metrics:

- `teacher_forced_loss_delta`
- `teacher_forced_perplexity_ratio` when available
- `teacher_forced_logit_max_abs_error`
- `teacher_forced_logit_mean_abs_error`
- `teacher_forced_logit_rmse`
- `teacher_forced_token_agreement_rate`
- `teacher_forced_target_match_rate`

`diagnostic` metrics:

- `replay_output_max_abs_error`
- shortlist recall metrics such as `shortlist_recall_exact_top_recall_mean`
- runtime breakdowns such as:
  - `execution_decode_shortlist_selection_ms_total`
  - `execution_decode_shortlist_materialization_ms_total`
  - `execution_decode_shortlist_candidate_approx_scoring_ms_total`
  - `execution_decode_backend_call_non_backend_ms_total`

Those families support different claims:

- a `systems` table supports throughput or memory claims
- a `quality` table supports fidelity claims
- a `diagnostic` table explains mechanisms and bottlenecks

They should only be collapsed into one paper figure when the rows come from the same held-out protocol and prompt set.

The minimum metadata schema for every promoted row should therefore be:

- model id and model family
- backend, device, and torch dtype
- layer profile or explicit routing config
- prompt family and dataset name
- split: `calibration` or `held_out`
- prompt count
- prompt length or context bucket
- batch size
- decode horizon or eval steps
- whether the row is `systems`, `quality`, or `diagnostic`

### 5.3. Current Artifact Coverage Against The Protocol

The current repo still only partially satisfies that standardized contract:

| Requirement | Current status | What is still missing |
| --- | --- | --- |
| exact-length serving harness | yes | repeat counts and confidence intervals should be reported by default |
| teacher-forced quality harness | yes | package it around named held-out prompt suites |
| shortlist recall / scorer diagnostics | yes | use them as supporting diagnostics rather than substitute evidence |
| split between calibration and held-out evidence | partial | current paper still relies on some calibration-style anecdotes |
| named datasets | partial | two synthetic four-prompt serving packs plus one small LongBench-derived QA mini-pack now exist, but broader suite coverage and quality-task counterparts are still missing |
| external baselines | partial | a StreamingLLM-style sink-plus-recent reference lane now exists on the Needle pack, but broader comparator coverage is still missing |
| one standard table shape across models | no | TinyLlama, SmolLM2, Qwen2.5, and Qwen3.5 still use mixed regimes |

That status table is the honest indicator of direction of travel. The project is no longer missing metrics entirely; it is missing consistency, held-out packaging, and external comparators.

### 5.4. TinyLlama 1.1B

TinyLlama remains the cleanest success case in the local profiles:

- prefill inspection at `577` tokens showed keys mostly staying conservative while values collapsed almost entirely to low-bit `M0`
- the best first-pass profile uses strict key routing in layers `3-21` with aggressive values
- the second pass, which made early key layers more aggressive, was a useful negative result: it barely changed KV memory, increased cost, and slightly worsened loss
- the best recorded teacher-forced loss delta in the local profile note is `+0.00008` with `1.0` agreement

This is good evidence that asymmetric K/V selection is useful. It is not yet enough evidence to claim that the specific TinyLlama profile generalizes.

### 5.5. SmolLM2 360M

SmolLM2 remains the strongest argument for cleaning up the bundle terminology:

- keys were genuinely mixed under the balanced probe
- values looked mostly tolerant overall, but late layers were fragile
- forcing late values into the `strict` bundle increased `M1` usage because `strict` means `M1 -> M0` on values
- the best current local checkpoint reduced late-key `M2` pressure and reached a recorded teacher-forced loss delta of `+0.0516` with `1.0` agreement

That result is interesting because it identifies a real planner interaction. It is also a warning that the current profiles are still artisanal.

### 5.6. Qwen2.5 3B: Selective Key Precision

Qwen2.5 3B is currently the clearest argument for explicit page-level or layer-head-level rescue:

- all-`M0` key pages fall to `0.25` greedy agreement at `4096`
- a selective exact-key policy that pins only `4.17%` of key pages to `M3` restores `1.00` agreement
- that selective policy stays close to the compressed memory regime: the current artifact reports `0.195x` KV bytes versus dense at `4096`, compared with `0.187x` for all-`M0` and `0.359x` for exact-K

This is stronger than the original PDF table because it removes the placeholder `"(est.)"` and makes the comparison explicit:

| Qwen2.5 3B @ 4096 | % key pages exact | Greedy agreement | KV bytes vs dense | Decode ms/step |
| --- | ---: | ---: | ---: | ---: |
| exact K (`K=M3`, `V=M0`) | `100%` | `1.00` | `0.359x` | `240.47` |
| all `M0` | `0%` | `0.25` | `0.187x` | `407.81` |
| selective exact K | `4.17%` | `1.00` | `0.195x` | `390.58` |

The important unresolved question is still whether this is genuinely page-level adaptivity or mostly a sparse set of layer/head overrides. Right now the latter story is better supported than the former.

### 5.7. Qwen3.5 0.8B: Hybrid Attention and Shortlisting

Qwen3.5 is the most useful evidence for the read-time selector because only a small subset of layers are full-attention layers. The latest dedicated rerun artifact in [`qwen35_cuda_shortlist_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl), summarized in [`performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md), gives a more mixed result than the earlier March 29 probe:

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Current read |
| ---: | ---: | ---: | ---: | --- |
| `4096` | `257.14` | `164.88` | `163.02` | shortlist win |
| `8192` | `167.94` | `167.37` | `171.77` | essentially flat |
| `16384` | `194.55` | `198.41` | `201.60` | shortlist slightly worse |

All nine rerun rows stayed on the same `per_kv_fallback` decode path. The newer instrumented large-context rerun narrowed the blocker further: for this Qwen3.5 CUDA serving lane, [`qwen35.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/qwen35.py) calls `decode_layer_torch(..., prefer_grouped_batching=hidden_states.device.type != "cuda")`, so grouped-batch validation and its rejection counters do not normally execute on CUDA for this path. The first forced-grouped ablation behind `DOTCACHE_QWEN35_FORCE_GROUPED_BATCHING=1` showed that grouped decode was not fundamentally dead on CUDA, but it was much worse for this workload and exposed `key_value_chunk_signature_mismatch` as the first concrete blocker. Two follow-up backend patches then removed that blocker and the next one, `key_signature_mismatch_across_groups`, strongly enough that the forced grouped path now runs end-to-end on the shortlist rows. The current systems lesson is therefore sharper than "grouped decode did not activate." On this lane, the default CUDA guard is still defensible today, but for a narrower reason: the grouped path is now operational and close to parity, not obviously broken.

The older March 29 probe in [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md) is still worth keeping as a historical result because it showed much larger gains through `16384`. But after the March 31 rerun, it should be treated as an encouraging earlier probe rather than the current paper table.

At larger context, the newest clean artifacts add a useful but mixed third datapoint. The serving read in [`qwen35_cuda_shortlist_large_context_repro_serving_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving_summary.md) now gives the shortlist rows as 3-run summaries with variance:

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | `n` |
| ---: | ---: | ---: | ---: | ---: |
| `32768` | `2312.64` one-off exact | `623.88 +/- 13.48` | `626.15 +/- 2.88` | `3` |
| `49152` | `3580.59` one-off exact | `741.45 +/- 26.85` | `792.68 +/- 31.76` | `3` |

But the quality-tail read is still not clean at `49152`. The underlying rows are now split more honestly:

- `32768` quality still comes from [`qwen35_cuda_shortlist_large_context_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl)
- `49152` quality is also preserved in the protocol-tagged held-out artifact [`qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_49152_heldout_quality_protocol.jsonl), which records `evaluation_split=held_out`, `evaluation_lane=quality`, `evaluation_prompt_count=1`, and `evaluation_protocol_version=2026-03-31`

| Context | Exact tail max abs logit error | Base shortlist | Layer-23 ctx |
| ---: | ---: | ---: | ---: |
| `32768` | `0.8984` | `3.5098` | `3.5137` |
| `49152` | `4.5742` | `7.0000` | `6.9648` |

The obvious follow-up was to widen the `49152` shortlist from `top_k=4` to `top_k=8`. That helped only modestly:

| `49152` config | Decode ms/step | Tail loss delta | Tail max abs logit error |
| --- | ---: | ---: | ---: |
| shortlist base, `top_k=4` | `752.13` | `+0.0130062` | `7.0000` |
| shortlist base, `top_k=8` | `819.41` serving / `793.73` quality | `+0.0113542` | `6.8711` |
| shortlist `layer:23` ctx, `top_k=8` | `893.25` serving / `1062.99` quality | `+0.0113542` | `6.8711` |

The first named benchmark-style pack is now also in place. The held-out serving artifact [`qwen35_cuda_needle_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_needle_pack_protocol_v1.jsonl), summarized in [`qwen35_cuda_needle_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_needle_pack_protocol_v1_summary.md), evaluates four fixed Needle-in-a-Haystack prompts at `32768` and `49152` for `exact`, `shortlist_base`, and `shortlist_l23_ctx`.

| Needle context | Case | `n` prompts | Retrieval accuracy | Exact-match rate | Mean decode ms/step | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `32768` | exact | `4` | `1.00` | `0.75` | `2496.10` | `316.83` |
| `32768` | shortlist base | `4` | `1.00` | `1.00` | `561.51` | `144.72` |
| `32768` | shortlist `layer:23` ctx | `4` | `1.00` | `1.00` | `509.53` | `26.05` |
| `49152` | exact | `4` | `1.00` | `0.75` | `3966.83` | `434.54` |
| `49152` | shortlist base | `4` | `1.00` | `0.75` | `759.82` | `176.56` |
| `49152` | shortlist `layer:23` ctx | `4` | `1.00` | `0.75` | `641.14` | `25.20` |

This is materially stronger than the earlier single-prompt Needle run because prompt count and variance are now visible. The current read is:

- retrieval correctness stayed perfect across all `24/24` rows
- shortlist keeps a large systems win on this named task family at both `32768` and `49152`
- strict exact-match is slightly weaker than retrieval correctness because the `shipment_token` prompt sometimes emits the correct token and then continues with `Question:`
- the earlier simple story that the layer-23 override is "just slower" is no longer right on this pack, but the prompt-sensitive variance is still too high to justify a confident default switch

The second fixed synthetic retrieval family is now also in place. The held-out serving artifact [`qwen35_cuda_passkey_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1.jsonl), summarized in [`qwen35_cuda_passkey_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1_summary.md), evaluates four fixed RULER-style passkey prompts at the same two contexts.

| Passkey context | Case | `n` prompts | Retrieval accuracy | Exact-match rate | Mean decode ms/step | 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `32768` | exact | `4` | `1.00` | `0.25` | `2390.28` | `200.40` |
| `32768` | shortlist base | `4` | `1.00` | `0.25` | `500.58` | `35.57` |
| `32768` | shortlist `layer:23` ctx | `4` | `1.00` | `0.25` | `511.36` | `40.69` |
| `49152` | exact | `4` | `1.00` | `0.25` | `3823.87` | `386.65` |
| `49152` | shortlist base | `4` | `1.00` | `0.25` | `662.52` | `62.46` |
| `49152` | shortlist `layer:23` ctx | `4` | `1.00` | `0.25` | `657.81` | `40.55` |

This second family is useful for a different reason than Needle:

- retrieval again stayed perfect across all `24/24` rows
- shortlist again kept a large systems win, roughly `4.7x` to `5.8x` faster than exact
- shortlist page counts stayed stable at `12240` for base and `12336` for the layer-23 override
- strict exact-match is much weaker than retrieval correctness because the model often emits the correct digits and then continues into prompt text such as `Question: What is` or `Vault record: the`
- the layer-23 override remains mixed rather than clearly promoted: it is slightly slower at `32768` and slightly faster at `49152`

The first non-synthetic named task family is now also in place. The held-out serving artifact [`qwen35_cuda_longbench_qa_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_longbench_qa_pack_protocol_v1.jsonl), summarized in [`qwen35_cuda_longbench_qa_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_longbench_qa_pack_protocol_v1_summary.md), evaluates four fixed LongBench QA rows with the official task prompts and official QA F1 scoring.

| LongBench QA case | `n` prompts | Mean QA F1 | Exact-match rate | Mean decode ms/step | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact | `4` | `0.14` | `0.00` | `743.41` | `337.48` |
| shortlist base | `4` | `0.08` | `0.00` | `178.55` | `12.77` |
| shortlist `layer:23` ctx | `4` | `0.08` | `0.00` | `176.41` | `17.30` |

This new family changes the benchmark read in an important way:

- it fills the most obvious benchmark-breadth gap with real benchmark rows rather than another handcrafted prompt pack
- shortlist still keeps a large systems win, about `4.2x` faster than exact on mean decode
- quality does not currently carry over cleanly: mean QA F1 drops from `0.14` under exact to `0.08` under both shortlist variants
- the per-row read is mixed rather than uniformly bad: `multifieldqa_en` improves slightly, `hotpotqa` degrades substantially, and `2wikimqa` plus `qasper` stay at `0.0` across all variants
- this is therefore not a benchmark-table win artifact; it is an honest signal that real-benchmark QA quality retention remains open

Operational note:

- the first LongBench wrapper pass exposed a real probe bug in [`run_qwen35_cuda_longbench_qa_probe.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_longbench_qa_probe.py): legitimate `longbench_row_index=0` rows were initially mis-handled
- that bug is now fixed on-branch, and the final canonical LongBench artifact has `12` rows and `0` error payloads

The first cheap external-style comparator on that same pack is now also available in [`qwen35_cuda_streaming_window_needle_pack_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1.jsonl), summarized in [`qwen35_cuda_streaming_window_needle_pack_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1_summary.md). This is a simple StreamingLLM-style sink-plus-recent reference lane with `256` sink tokens, `1024` recent tokens, and no query-aware shortlist expansion.

| Comparator context | Case | Retrieval accuracy | Mean decode ms/step | Read |
| ---: | --- | ---: | ---: | --- |
| `32768` | exact | `1.00` | `2521.60` | high-quality, expensive |
| `32768` | streaming sink+recent | `0.00` | `156.65` | very fast, unusable retrieval |
| `32768` | shortlist base | `1.00` | `474.23` | slower than streaming, far faster than exact |
| `49152` | exact | `1.00` | `3909.29` | high-quality, expensive |
| `49152` | streaming sink+recent | `0.00` | `188.55` | very fast, unusable retrieval |
| `49152` | shortlist base | `1.00` | `629.46` | slower than streaming, far faster than exact |

This comparator sharpens the systems claim in the right direction:

- a cheap sink-plus-recent reference can be dramatically faster than DotCache shortlist
- on this Needle pack it also fails retrieval completely, emitting color prefixes like `crimson`, `amber`, `cobalt`, and `silver` instead of the planted tokens
- DotCache shortlist therefore looks less like "the only thing compared against exact" and more like a middle point on a visible speed-quality frontier

The other obvious follow-up was to force grouped batching on CUDA and see whether the default guard was merely hiding a faster path. The first answer was "not yet": the early forced-grouped serving artifact in [`qwen35_cuda_shortlist_large_context_probe_forced_grouped.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe_forced_grouped.jsonl) shows partial grouped activation but much worse throughput:

| Forced-grouped context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Decode-path read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `2309.26` | `1916.62` | `1843.35` | mixed grouped + fallback |
| `49152` | `3643.76` | `2116.18` | `2059.90` | mixed grouped + fallback |

That forced run is still useful because it answers two systems questions cleanly:

- grouped batching can execute on this CUDA lane when forced
- the dominant rejection reason is now concrete: `key_value_chunk_signature_mismatch`

After the chunk-schedule split and mixed-signature bucketing fixes, the follow-up forced-grouped artifact in [`qwen35_cuda_shortlist_large_context_probe_forced_grouped_bucketed.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe_forced_grouped_bucketed.jsonl) is materially stronger:

| Forced-grouped bucketed context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Decode-path read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `2335.11` | `716.36` | `693.11` | all successful rows `grouped_batched=24, per_kv_fallback=0` |
| `49152` | `3658.27` | `777.28` | `766.36` | all rows `grouped_batched=24, per_kv_fallback=0` |

The `32768 shortlist_base` row above comes from the direct one-off rerun in [`qwen35_cuda_shortlist_32768_forced_grouped_bucketed_base_single.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32768_forced_grouped_bucketed_base_single.jsonl), because the wrapper matrix had a `NoExactRow` capture miss for that case. That looks like a wrapper bookkeeping issue rather than a backend failure, but it should still be recorded explicitly.

This newer grouped result changes the systems read in an important way:

- grouped decode is now operational end-to-end on this CUDA shortlist lane
- the previous grouped blocker `key_signature_mismatch_across_groups` disappears on the successful rows
- forced grouped shortlist throughput is now close to the default non-forced path
- it still does not clearly beat the default path, so the default CUDA guard remains acceptable as a performance choice rather than as a correctness workaround

Two more follow-ups now close the loop on that grouped path. First, the forced-grouped quality-tail artifact in [`qwen35_cuda_shortlist_large_context_quality_tail_forced_grouped.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail_forced_grouped.jsonl) shows that grouped decode is fully active and quality-stable on all six rows. It does not fix the `49152` loss-tail issue, but it also does not materially worsen it relative to the earlier non-forced shortlist read. Second, the 3x serving reproducibility pass in [`qwen35_cuda_shortlist_large_context_repro_serving`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving) shows that grouped decode is repeatable but not a reproducible overall win: three of the four shortlist cases are still slower under forced grouped by about `4.5%` to `7.4%`, while only `49152 shortlist_l23_ctx` edges ahead and only by `0.47%`.

That makes the current grouped-CUDA conclusion much cleaner than before:

- grouped decode is now real, repeatable, and quality-stable on this lane
- grouped decode is not the missing fix for the `49152` quality regime
- grouped decode still does not justify replacing the default CUDA shortlist path on performance grounds alone

So the current paper claim should be: shortlist has a genuine large-context serving-speed signal, but the long-context quality story is still unresolved, widening the shortlist helps only a little, and the layer-23 context-aware widening does not materially solve it. The cleaned-up note in [`qwen35_cuda_shortlist_paper_table.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_paper_table.md) now separates all of these CUDA reads explicitly and uses the repro summary / held-out quality artifacts rather than only one-off journal prose.

### 5.8. What The Current Evidence Actually Supports

At this point the paper can make four concrete claims without overreaching.

1. Layer-aware routing is necessary.
Different models and tensor kinds really do want different policies. TinyLlama, SmolLM2, Qwen2.5, and Qwen3.5 all expose different failure modes under uniform routing.

2. Query-aware shortlisting is a real systems lever.
On the Qwen3.5 CUDA lane, shortlist execution produces substantial serving-speed wins once context reaches `32768+`. The default serving integration still runs those rows through `per_kv_fallback`, but the forced-grouped follow-ups now show that grouped CUDA can execute end-to-end with comparable quality and near-parity serving throughput. The fixed four-prompt Needle and passkey packs both preserve retrieval at `32768` and `49152` while keeping those large serving wins, and the LongBench-derived QA mini-pack still shows about a `4.2x` systems win on real benchmark rows even though QA F1 retention is not yet good enough. The streaming-window comparator then shows that an even faster simple reference baseline collapses retrieval entirely on the same Needle pack.

3. Long-context quality is now the binding problem.
The systems bottleneck is no longer "can we cut the attended page set?" The harder question is "how do we preserve quality once we do?" The `49152` tail results make that explicit, and the LongBench QA mini-pack now shows the same issue on real benchmark rows rather than only on synthetic probes.

4. Cheap rescue heuristics and backend-path flips are not enough yet.
Both the layer-23 context-aware widening and the `top_k=8` follow-up improve the story only marginally. Switching the backend path to grouped decode also does not fix the `49152` loss tail or produce a stable serving win. These are useful diagnostics, not final fixes.

What the paper should **not** claim yet:

- that DotCache shortlist is a stable throughput win at every tested context
- that the `49152` configuration is quality-clean
- that grouped-batched decode is already the right default on the main CUDA lane
- that page-level observed-stat routing has already beaten simpler fixed-policy baselines on a multi-prompt standardized benchmark suite

What this does **not** yet show:

- end-to-end TTFT
- batch scaling
- selector overhead inside the serving loop
- a broader standardized long-context suite beyond these first three small families, such as a fuller LongBench sweep, a fuller RULER integration, or named quality-task counterparts

## 6. Related Work and Novelty Positioning

The reviewed PDF under-positioned DotCache against nearby work. The right framing is not that DotCache invented every ingredient. It did not. The novelty is the integration of several ingredients inside one compressed-domain execution story.

### 6.1. Closest Read-Time Selector Analogs

- [StreamingLLM](https://arxiv.org/abs/2309.17453) established the sink-plus-recent-window pattern that DotCache also uses in its shortlist.
- [Quest](https://arxiv.org/abs/2406.10774) is the closest direct neighbor on read-time page selection because it also uses query-aware page retrieval with per-page extrema metadata.
- [PQCache](https://arxiv.org/abs/2407.12820) is another adjacent retrieval-first view of long-context KV access.

DotCache differs by making read-time selection one part of a larger compressed-domain page format and execution contract rather than the entire proposal.

### 6.2. Layer-Aware Budgeting and Hybrid Compression

- [PyramidKV](https://arxiv.org/abs/2406.02069) and [SqueezeAttention](https://arxiv.org/abs/2404.04793) both argue that layer budgets should differ rather than treating all layers equally.
- [TailorKV](https://arxiv.org/abs/2505.19586) is especially relevant because it also exploits layer heterogeneity, although through a quantization-plus-offloading hybrid instead of compressed-domain page execution.

DotCache should therefore claim layer-aware routing as a necessary subsystem for its execution model, not as a wholly unprecedented observation.

### 6.3. Mixed-Precision Rescue Paths

- [No Token Left Behind](https://arxiv.org/abs/2402.18096) is relevant because it preserves important KV states at higher precision through importance-aware mixed precision.
- [KIVI](https://arxiv.org/abs/2402.02750) already argued that keys and values benefit from asymmetric treatment.
- [QServe](https://arxiv.org/abs/2405.04532) and [InnerQ](https://arxiv.org/abs/2602.23200) are the clearest execution-side motivation for keeping decode close to the actual attention work.

The right novelty sentence for this paper is therefore:

> DotCache combines layer-aware write-time codec routing, explicit high-precision rescue pages, and query-aware read-time shortlisting inside a compressed-domain attention substrate, instead of treating compression, rescue, and retrieval as separate systems.

## 7. What Still Needs to Happen Before Submission

The review was right that the current draft is not publication-ready. The remaining work is not cosmetic.

### 7.1. Use The Standardized Protocol As The Mainline Contract

The protocol above is now specific enough to serve as the paper's evaluation contract. What still needs to happen is not to invent another protocol, but to run the mainline tables under it consistently.

Before submission, every promoted result should therefore include:

- named dataset or prompt suite
- prompt count
- hardware details
- batch size
- decode horizon or eval horizon
- variance or confidence intervals
- an explicit `calibration` versus `held_out` label

The biggest remaining gap is that the current artifact still mixes calibration anecdotes with held-out-style rows. The next draft should separate those cleanly.

### 7.2. Add Baselines That Match the Claim

The current artifact is still stronger on internal comparisons than on external ones. A publication draft should eventually add external baselines from both sides of the problem, but the first cheap reference lane is now in place:

- a StreamingLLM-style sink-plus-recent reference baseline on the Needle pack

What is still missing:

- low-bit KV quantization baselines
- query-aware retrieval / sparse attention baselines
- layer-budget baselines
- rematerialization or offloading baselines when relevant

### 7.3. Isolate the Ingredients

The paper also needs ablations for:

- K/V asymmetry
- recent-window escape
- write-time routing vs fixed per-layer policy
- shortlist windows vs relevance gating
- sketch vs envelope scoring
- selector implementation overhead
- page size and group size

The current results suggest one priority ordering rather than a broad sweep:

1. one stronger `49152` quality rescue that is not just another `top_k` increase, because grouped decode has now been tested and is not the missing fix
2. add selector / score / mix timing splits on the grouped CUDA path, because the remaining grouped/default gap is now small enough to profile directly
3. only then a wider ablation grid

### 7.4. Make the Profile Discovery Scientific

Right now profile discovery is still manual enough that a reviewer can reasonably call it hand-tuning. A credible next draft needs one of:

- an automated search procedure over layer bundles
- a calibration/test split with frozen policies on held-out prompts
- a small learned router trained only on calibration traces

## 8. Conclusion

The reconstructed artifact already contains a real systems idea: compressed-domain KV execution needs an explicit selection layer. The code and artifacts support that claim. They also support the review's criticism that the earlier draft overreached when it presented heuristic, mixed-protocol evidence as though it were already a benchmark-complete conference result.

The strongest current result is not "DotCache wins everywhere." It is more specific and more credible: layer-aware shortlisting can create genuine long-context serving-speed wins, but quality recovery becomes the new hard problem, and simple shortlist widening does not solve it. That is a worthwhile systems story, but only if the paper says it plainly.

The next draft should therefore keep the core idea, tighten the terminology, state selector complexity honestly, foreground the mixed `32768/49152` outcome, and separate what is already demonstrated from what still needs standardized evaluation. That makes the paper narrower, but also much stronger.
