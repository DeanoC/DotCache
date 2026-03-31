# Layer-Aware Page Selection in DotCache

_Working manuscript source reconstructed from `dotcache_layer_page_selection.pdf` on 2026-03-31._

## Abstract

DotCache proposes executing attention directly on page-organized, low-bit key-value (KV) caches rather than dequantizing compressed tokens back to full precision before the attention kernel runs. A critical but underexplored part of that design is the selection subsystem that decides which codec, bitwidth, and execution mode to assign to each page. This note describes the current DotCache selection stack: write-time page routing, recent-window escape, and read-time shortlist gating. The current artifact supports six page modes (`M0`, `M1`, `M2`, `M3`, `M4`, `T3`) and has been exercised on TinyLlama, SmolLM2, Qwen2.5, and Qwen3.5. The main engineering conclusion is that layer-aware selection is necessary for compressed-domain execution to remain usable. The main scientific caveat is equally important: the present evidence is still prototype-grade. The selector design is concrete and the failure modes are real, but the empirical story still needs standardized benchmarks, calibration/test splits, and stronger baselines before this can be framed as a publication-ready systems paper.

## 1. Introduction

DotCache treats low-bit KV representations as an execution format rather than a storage format. Instead of compressing keys and values for memory savings and then widening them back to full precision before attention, DotCache tries to keep decode fused with score and mix computation. That design targets the widening and metadata traffic that recent serving work has identified as a dominant overhead in low-bit pipelines.

Compressed-domain execution creates a systems question that storage-only quantization papers can mostly sidestep: which codec should each page use, and when should the runtime bypass compression entirely? In DotCache, the answer propagates into the attention kernel itself because the kernel consumes compressed pages directly. A bad per-layer or per-page choice is not normalized away by a generic dequantization stage later.

This draft focuses on that selection layer: the write-time policy that maps observed page statistics to a page mode, the recent-window escape that keeps the hottest tokens in high precision, and the read-time shortlist that limits which old pages are attended. The core claim is modest and specific. DotCache is interesting because it combines compressed-domain execution with adaptive page routing. The current artifact does not yet prove that this combination beats the best long-context baselines on standard benchmark suites.

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

### 5.1. Current Protocol Snapshot

The artifact is currently a mix of four evaluation lanes rather than one standardized benchmark suite:

| Model family | Backend / hardware | Current prompt regime | Metrics reported today |
| --- | --- | --- | --- |
| TinyLlama 1.1B | local MPS | prefill at `577`, teacher-forced windows around `320/288/16` | page-mode counts, loss delta, token agreement |
| SmolLM2 360M | local MPS | prefill at `1024`, teacher-forced windows around `1032/1024/8` | page-mode counts, loss delta, token agreement |
| Qwen2.5 3B | CUDA / RTX 5090 lane | exact prompt lengths `1024`, `2048`, `4096` | greedy agreement, KV memory ratio, decode ms/step |
| Qwen3.5 0.8B | CUDA / RTX 5090 lane | serving at `4096`, `8192`, `16384`, `4` decode tokens | ms/step, tok/s, selected/candidate pages, replay/logit error |

The common prompt-construction pattern for several harnesses is a repeated synthetic unit string (`"Cache locality matters for fast decoding."`) trimmed to exact token length. That is fine for development, but a publication draft needs named datasets, prompt counts, and variance.

### 5.2. Metric Definitions

The paper should define its metrics explicitly:

- `loss delta`: teacher-forced cross-entropy loss difference between DotCache and the dense or exact-reference run on the same prompt slice
- `token agreement`: greedy next-token agreement rate over the checked decode horizon
- `replay_output_max_abs_error`: maximum absolute deviation in replayed layer output tensors against the reference run
- `teacher_forced_logit_max_abs_error`: maximum absolute deviation in logits under teacher forcing

Those metrics are not interchangeable. A cleaned-up paper should stop placing them in the same summary table unless they were collected under the same protocol.

### 5.3. TinyLlama 1.1B

TinyLlama remains the cleanest success case in the local profiles:

- prefill inspection at `577` tokens showed keys mostly staying conservative while values collapsed almost entirely to low-bit `M0`
- the best first-pass profile uses strict key routing in layers `3-21` with aggressive values
- the second pass, which made early key layers more aggressive, was a useful negative result: it barely changed KV memory, increased cost, and slightly worsened loss
- the best recorded teacher-forced loss delta in the local profile note is `+0.00008` with `1.0` agreement

This is good evidence that asymmetric K/V selection is useful. It is not yet enough evidence to claim that the specific TinyLlama profile generalizes.

### 5.4. SmolLM2 360M

SmolLM2 remains the strongest argument for cleaning up the bundle terminology:

- keys were genuinely mixed under the balanced probe
- values looked mostly tolerant overall, but late layers were fragile
- forcing late values into the `strict` bundle increased `M1` usage because `strict` means `M1 -> M0` on values
- the best current local checkpoint reduced late-key `M2` pressure and reached a recorded teacher-forced loss delta of `+0.0516` with `1.0` agreement

That result is interesting because it identifies a real planner interaction. It is also a warning that the current profiles are still artisanal.

### 5.5. Qwen2.5 3B: Selective Key Precision

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

### 5.6. Qwen3.5 0.8B: Hybrid Attention and Shortlisting

Qwen3.5 is the most useful evidence for the read-time selector because only a small subset of layers are full-attention layers. The latest dedicated rerun artifact in [`qwen35_cuda_shortlist_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl), summarized in [`performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md), gives a more mixed result than the earlier March 29 probe:

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Current read |
| ---: | ---: | ---: | ---: | --- |
| `4096` | `257.14` | `164.88` | `163.02` | shortlist win |
| `8192` | `167.94` | `167.37` | `171.77` | essentially flat |
| `16384` | `194.55` | `198.41` | `201.60` | shortlist slightly worse |

All nine rerun rows stayed on the same `per_kv_fallback` decode path. That is the current systems lesson: shortlist execution is operational on CUDA, but the expected grouped-batched speed path still is not activating on this lane, so long-context speedups are not yet stable.

The older March 29 probe in [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md) is still worth keeping as a historical result because it showed much larger gains through `16384`. But after the March 31 rerun, it should be treated as an encouraging earlier probe rather than the current paper table.

At larger context, the newest clean wrapper artifacts add a useful but mixed third datapoint. The serving rerun in [`qwen35_cuda_shortlist_large_context_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe.jsonl) shows that shortlist is a real decode-speed win at both `32768` and `49152`:

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step |
| ---: | ---: | ---: | ---: |
| `32768` | `2298.36` | `673.12` | `671.80` |
| `49152` | `3675.23` | `786.35` | `844.04` |

But the quality-tail rerun in [`qwen35_cuda_shortlist_large_context_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl) is not clean at `49152`:

| Context | Exact tail max abs logit error | Base shortlist | Layer-23 ctx |
| ---: | ---: | ---: | ---: |
| `32768` | `0.8984` | `3.5098` | `3.5137` |
| `49152` | `4.5742` | `7.0000` | `6.9648` |

The obvious follow-up was to widen the `49152` shortlist from `top_k=4` to `top_k=8`. That helped only modestly:

| `49152` config | Decode ms/step | Tail loss delta | Tail max abs logit error |
| --- | ---: | ---: | ---: |
| shortlist base, `top_k=4` | `786.35` | `+0.0130062` | `7.0000` |
| shortlist base, `top_k=8` | `819.41` serving / `793.73` quality | `+0.0113542` | `6.8711` |
| shortlist `layer:23` ctx, `top_k=8` | `893.25` serving / `1062.99` quality | `+0.0113542` | `6.8711` |

So the current paper claim should be: shortlist has a genuine large-context serving-speed signal, but the long-context quality story is still unresolved, widening the shortlist helps only a little, and the layer-23 context-aware widening does not materially solve it. The cleaned-up note in [`qwen35_cuda_shortlist_paper_table.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_paper_table.md) now separates all of these CUDA reads explicitly.

What this does **not** yet show:

- benchmark variance
- end-to-end TTFT
- batch scaling
- selector overhead inside the serving loop
- standardized long-context tasks like LongBench, RULER, or Needle-in-a-Haystack

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

### 7.1. Standardize the Empirical Protocol

Before submission, the paper needs one evaluation table shape with:

- named datasets
- prompt counts
- hardware details
- batch size
- decode horizon
- variance or confidence intervals

At minimum, that means separating:

- local heuristic profile discovery
- calibration runs
- held-out benchmark runs

### 7.2. Add Baselines That Match the Claim

The current artifact is strongest on internal comparisons. A publication draft needs external baselines from both sides of the problem:

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

### 7.4. Make the Profile Discovery Scientific

Right now profile discovery is still manual enough that a reviewer can reasonably call it hand-tuning. A credible next draft needs one of:

- an automated search procedure over layer bundles
- a calibration/test split with frozen policies on held-out prompts
- a small learned router trained only on calibration traces

## 8. Conclusion

The reconstructed artifact already contains a real systems idea: compressed-domain KV execution needs an explicit selection layer. The code and artifacts support that claim. They also support the review's criticism that the current paper overreaches when it presents heuristic, mixed-protocol evidence as though it were already a benchmark-complete conference result.

The next draft should keep the core idea, tighten the terminology, state selector complexity honestly, and separate what is already demonstrated from what still needs standardized evaluation. That makes the paper narrower, but also much stronger.
