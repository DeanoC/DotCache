# Layer-Aware Page Selection in DotCache: A Work in Progress

*DotCache Project — March 2026*

---

DotCache is an experiment in compressed-domain KV-cache execution for LLM inference. The idea is to keep attention running directly on low-bit compressed pages rather than decompressing them back to full precision before each decode step. That sounds straightforward, but it creates a problem that most quantization papers can sidestep: *which codec should each page use, and when should the runtime skip compression entirely?* This post describes where we are with that selection subsystem.

## The Setup

A DotCache page is the atomic unit of compressed KV storage, indexed by *(layer, head, token range, kind)* where kind is key or value. Within a page, vectors are split into 32-channel groups. Each group gets encoded with one of six modes:

| Mode | What it does | Where it applies |
|---|---|---|
| M0 | Affine quantization (scale + bias) | Default workhorse at 2–4 bits, K and V |
| M1 | Nonuniform LUT quantization | Cluster-friendly value path |
| M2 | Sketch-plus-residual | Inner-product-aware key scoring |
| M3 | High-precision escape | Recent tokens and explicit rescue pages |
| M4 | Projected-basis key encoding | Structured key compression |
| T3 | TurboQuant-style vector quant | Experimental aggressive 3-bit |

The selection question is: given a freshly produced page of keys or values, which mode and bitwidth keep attention quality acceptable without blowing the memory and runtime budget? This isn't a global quantization decision — the right answer depends on the producing layer, the tensor kind (keys vs. values behave very differently), how old the tokens are, and the measured statistics of the page itself.

## Write-Time Selection: Policy Bundles

At write time, the planner observes four page statistics (RMS, absolute max, outlier fraction, and mean channel range) and cascades through a mode candidate list until something fits. The candidate lists are organized into four named *policy bundles*:

| Bundle | Key candidates | Value candidates |
|---|---|---|
| exact | Default mode only — no adaptive search | Default mode only |
| strict | M0/4b | M1/4b → M0/4b |
| balanced | M0/2b → M2 or M4 → M0/4b | M0/3b → M1/4b → M0/4b |
| aggressive | Same as balanced with relaxed thresholds | M0/2b → M0/3b → M1/4b → M0/4b |

One important thing we got wrong in the original draft: these are *not* a monotonic sensitivity ladder. In particular, `strict` is not uniformly more conservative than `balanced` for value pages — the strict value bundle tries M1/4b before M0/4b, while balanced starts with M0/3b and may avoid M1 entirely. That distinction caused real quality problems on SmolLM2 and is why we're careful about the wording now.

Pages younger than `recent_window` tokens get routed to M3 high-precision escape *before* bundle evaluation. This is a separate mechanism from the bundle labels.

## Read-Time Selection: Query-Aware Shortlisting

At read time during decode, DotCache runs a selector that limits which old pages get attended at all. It combines:

- A **sink window** (256 tokens) for early positional anchors
- A **recent window** (1024–4096 tokens depending on context length) for the newest pages
- **Relevance gating** over the old middle region, using either sketch scoring (compact mean vector per page) or envelope scoring (per-channel min/max bounds)

The top-K most relevant old pages are included; everything else is skipped. The important complexity note: the selector itself still scores every candidate page linearly, so the correct claim is *sub-linear selected-set attention*, not globally sub-linear decode. The selector overhead is small (~0.016ms per call at 49K context) but not zero, and naive array copying can dominate it — view-like or packed-span returns are meaningfully faster.

## What the Current Evidence Shows

We've run this on four model families. Here's what each one is teaching us.

**TinyLlama 1.1B** is the cleanest success case. Keys stay conservative, values collapse almost entirely to low-bit M0. The best profile (strict keys in layers 3–21, aggressive values) reaches a teacher-forced loss delta of +0.00008 with 1.0 token agreement. A second pass that made early key layers more aggressive was a useful negative: it barely changed KV memory and slightly worsened loss. Asymmetric K/V selection works, and conservative mid-stack key layers are load-bearing.

**SmolLM2 360M** is the best argument for getting the bundle terminology right. Forcing late-layer values into the strict bundle *increased* M1 usage (because strict means M1→M0 on values) and degraded loss. The fixed profile reached +0.0516 loss delta with 1.0 agreement. The real lesson is that current profiles are artisanal — the planner interactions aren't obvious without per-model probing.

**Qwen2.5 3B** makes the clearest argument for explicit page-level rescue. All-M0 key pages fall to 0.25 greedy agreement at 4096 tokens. Pinning just 4.17% of key pages to exact M3 (two specific layer/head combinations) restores 1.00 agreement at minimal memory cost:

| Config | % K Exact | Agreement | KV vs Dense | ms/step |
|---|---:|---:|---:|---:|
| Exact K (K=M3, V=M0) | 100% | 1.00 | 0.359x | 240.47 |
| All M0 | 0% | 0.25 | 0.187x | 407.81 |
| Selective exact K | 4.17% | 1.00 | 0.195x | 390.58 |

**Qwen3.5 0.8B** is where the read-time selector gets interesting — and honest. Qwen3.5 has a hybrid attention architecture where only six of its layers use full attention (layers 3, 7, 11, 15, 19, 23). That makes the selector's effect both larger and more focused.

## The Headline Speed Numbers

At 32K and 49K tokens, shortlisting is a substantial serving-speed win. These are 3-run means with variance on the shortlist rows:

| Context | Exact ms/step | Shortlist ms/step | L-23 ctx ms/step | Speedup |
|---:|---:|---:|---:|---:|
| 32,768 | 2312.64 | 623.88 ± 13.48 | 626.15 ± 2.88 | 3.7x |
| 49,152 | 3580.59 | 741.45 ± 26.85 | 792.68 ± 31.76 | 4.8x |

## Needle-in-a-Haystack: Where It Works

The retrieval results on a fixed four-prompt Needle-in-a-Haystack pack give the clearest win:

| Context | Case | Retrieval | Exact Match | ms/step | 95% CI |
|---:|---|---:|---:|---:|---:|
| 32,768 | exact | 1.00 | 0.75 | 2496.10 | 316.83 |
| 32,768 | shortlist | 1.00 | 1.00 | 561.51 | 144.72 |
| 32,768 | shortlist L23 | 1.00 | 1.00 | 509.53 | 26.05 |
| 49,152 | exact | 1.00 | 0.75 | 3966.83 | 434.54 |
| 49,152 | shortlist | 1.00 | 0.75 | 759.82 | 176.56 |
| 49,152 | shortlist L23 | 1.00 | 0.75 | 641.14 | 25.20 |

Retrieval accuracy stayed perfect across all 24 rows. Shortlisting keeps a 4.4–6.2x speed win while fully preserving the model's ability to find the needle. A RULER-style passkey pack shows the same pattern: perfect retrieval across all 24 rows, 4.7–5.8x speedup.

## The Comparator That Puts It in Context

The natural question is: how does this compare to a simple baseline? We ran a StreamingLLM-style sink-plus-recent reference (256 sink tokens, 1024 recent, no query-aware expansion) on the same Needle pack:

| Context | Case | Retrieval | ms/step |
|---:|---|---:|---:|
| 32,768 | exact | 1.00 | 2521.60 |
| 32,768 | streaming sink+recent | 0.00 | 156.65 |
| 32,768 | shortlist | 1.00 | 474.23 |
| 49,152 | exact | 1.00 | 3909.29 |
| 49,152 | streaming sink+recent | 0.00 | 188.55 |
| 49,152 | shortlist | 1.00 | 629.46 |

The streaming baseline is faster than DotCache shortlist — but it completely collapses retrieval on every prompt, emitting color prefixes like *crimson*, *amber*, *cobalt* instead of the planted tokens. DotCache shortlist sits on a visible speed–quality frontier between "cheap and broken" and "correct and expensive." That's the position we want to occupy, and now we have numbers to show it.

## The Open Problem: LongBench QA

Here's where it gets less comfortable. The first non-synthetic named benchmark — a four-prompt LongBench QA mini-pack with official F1 scoring — does not go well:

| Case | Mean QA F1 | ms/step |
|---|---:|---:|
| exact | 0.14 | 743.41 |
| shortlist base | 0.08 | 178.55 |
| shortlist L23 ctx | 0.08 | 176.41 |

Shortlisting is 4.2x faster, but QA F1 drops from 0.14 to 0.08. A rescue matrix testing top-K=8, a quality-biased shortlist profile, and layer-23 widening all showed no improvement. We dug into the worst case (HotpotQA) and found that the same old page ranges keep missing shortlist selection over and over: ranges 1296:1312, 5824:5840, 1376:1392, 624:640, and 8800:8816 are repeatedly absent. The next quality fix is not another generic knob — it's understanding why those specific pages keep getting dropped and fixing the selection behavior directly.

One important caveat: Qwen3.5 0.8B's exact baseline QA F1 of 0.14 is itself low. Whether the drop to 0.08 represents a compression problem or a model capacity issue isn't fully separated yet.

## The 49K Quality Tail

The synthetic quality metrics also still flag an unresolved problem. Teacher-forced logit max absolute error at 49K tokens:

| Context | Exact | Shortlist Base | L-23 ctx |
|---:|---:|---:|---:|
| 32,768 | 0.90 | 3.51 | 3.51 |
| 49,152 | 4.57 | 7.00 | 6.96 |

Widening to top-K=8 helped only slightly (7.0 → 6.87). We also investigated whether enabling grouped CUDA batching — which was disabled by default — might be the missing fix. After implementing chunk-schedule splitting and mixed-signature bucketing, the grouped path now runs end-to-end with all rows taking the grouped path and near-parity throughput (716 ms/step vs. 624 for the default). But it doesn't fix the quality tail, and it's 4.5–7.4% slower in most shortlist cases. The conclusion: grouped decode is now operational and repeatable, but it's not the fix we were looking for.

## Four Things the Evidence Actually Supports

1. **Layer-aware routing is necessary.** All four model families expose different failure modes under uniform routing. There's no single codec assignment that works everywhere.

2. **Query-aware shortlisting is a real systems lever.** On the Qwen3.5 CUDA lane, shortlist gives 3.7–4.8x speed wins at 32K+ tokens. The Needle and passkey packs show perfect retrieval is maintained. The streaming comparator shows that simpler baselines can be faster but retrieval-broken.

3. **Long-context quality is the binding problem.** The 49K quality tail and the LongBench QA F1 drop are the same issue from different angles. The diagnostic trail now points to specific repeatedly-missed pages as the root cause.

4. **Cheap rescue heuristics don't solve it.** Top-K widening, layer-specific overrides, and grouped CUDA decode all improve the story marginally at most. The fix needs to be in selection behavior itself.

## What's Still Missing

The honest list: end-to-end TTFT, batch scaling, selector-plus-attention timing decomposition inside the actual serving loop, competitive external baselines (Quest, H2O), broader benchmark coverage beyond these three small packs, and automated profile discovery. The current profiles are still hand-tuned per model.

## Where This Is Headed

The core design — write-time codec routing, high-precision escape, and read-time shortlisting inside a compressed-domain execution substrate — is sound and demonstrably necessary. The next concrete step is fixing the specific page selection misses identified by the HotpotQA diagnostic. After that: proper Quest/H2O comparators, larger models, and turning the three small benchmark packs into something statistically defensible.

The systems story is real. The quality story is still open. That's where we are.

---

*Code and artifacts: [DotCache project repository]*
