# Performance Journal

This file is the high-signal running summary of what we tried, what moved, and what did not.

Raw append-only run history lives in [benchmarks/results/history.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/history.jsonl).
The latest targeted envelope sweep lives in [envelope_sweep_4k.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/envelope_sweep_4k.jsonl).
The latest long-context tuner output lives in [envelope_tuner_8k_16k.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/envelope_tuner_8k_16k.jsonl).

## Current Status

Current branch head: `f851870`

### First NVIDIA CUDA snapshot

The new eager `torch_cuda` backend now runs on this NVIDIA RTX 2000 Ada machine through the same shared torch accelerator path as `torch_mps`.

First decode-step ladder on `torch_cuda`:

| Context | Prepare ms | Decode ms | Host-to-Device Bytes | Max Abs Logit Error | Max Abs Output Error |
|---|---:|---:|---:|---:|---:|
| `64` | `7.22` | `2.31` | `10,240` | `8.58e-06` | `7.15e-07` |
| `256` | `0.35` | `3.10` | `40,960` | `1.24e-05` | `4.05e-06` |
| `1024` | `0.66` | `6.86` | `163,840` | `1.10e-05` | `2.62e-06` |
| `4096` | `2.38` | `21.50` | `655,360` | `1.53e-05` | `7.15e-07` |

That is the right first microbench read of the CUDA port:

- the shared torch accelerator refactor is numerically stable on CUDA
- exact compressed-domain decode runs with `0` execution-time host-to-device bytes once pages are prepared
- latency scales cleanly with context in the current eager path
- the remaining work is performance work, not correctness rescue

First exact session-shaped CUDA checkpoint at `4096` context:

| Backend | Preload ms | Append ms/step | Decode ms/step | Session ms/step | Decode H2D Bytes/Step | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|
| `torch_cuda` | `10.47` | `0.39` | `33.11` | `33.50` | `0` | `2.15e-06` |

Compared with the CPU reference on the same benchmark:

- CPU decode landed at `14.28 ms/step`
- CUDA exact DotCache decode landed at `33.11 ms/step`
- the current eager CUDA path is still about `0.43x` of CPU decode speed on this synthetic exact session benchmark
- append is already cheap, and decode upload churn is gone; the main remaining gap is the eager page kernel itself

Latest tiny-random LLaMA smoke run on CUDA:

| Backend | Prompt Len | Decode Steps | Decode ms/step | Dense Decode ms/step | Greedy Agreement | Teacher-Forced Max Abs Logit Drift |
|---|---:|---:|---:|---:|---:|---:|
| `torch_cuda` | `6` | `3` | `17.65` | `4.24` | `1.00` | `9.79e-05` |

That confirms the model path is working correctly on CUDA:

- the Phase 5 LLaMA harness runs end to end on `cuda`
- greedy agreement stayed exact over the tested steps
- decode and prefill ingest both ran with `0` host-to-device bytes after the on-device torch path was selected

First real TinyLlama dense-vs-DotCache CUDA comparison:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Dense Final KV Bytes | DotCache Resident Bytes | DotCache/Dense KV Ratio | Greedy Agreement |
|---|---:|---:|---:|---:|---:|---:|
| `10` | `40.17` | `72.61` | `585,728` | `5,767,168` | `9.85x` | `1.00` |
| `289` | `22.31` | `132.15` | `13,156,352` | `7,569,408` | `0.58x` | `1.00` |
| `577` | `23.49` | `134.48` | `26,132,480` | `9,371,648` | `0.36x` | `1.00` |

That is the honest first CUDA model-level read:

- the CUDA port is functionally correct on a real TinyLlama checkpoint with full greedy agreement at all tested prompt lengths
- dense CUDA KV is still much faster on decode than the current eager DotCache path
- the KV-memory crossover is already visible by `289` prompt tokens, where DotCache drops to about `58%` of dense KV bytes
- by `577` prompt tokens, DotCache is down to about `36%` of dense KV bytes while still trailing on decode latency
- short-prompt overhead is still dominated by resident prepared-tail machinery, so DotCache is much worse than dense on KV bytes at `10` prompt tokens
- the first CUDA pass is therefore a successful parity port, not yet a performance win

One operational note from the first real-model CUDA run:

- the initial TinyLlama download hit a Hugging Face `429` rate limit once and then recovered on retry
- the benchmark still completed and was recorded successfully in the history log

First higher-context CUDA frontier probe on SmolLM2 360M:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Dense Final KV Bytes | DotCache Resident Bytes | DotCache/Dense KV Ratio | Status |
|---|---:|---:|---:|---:|---:|---|
| `2048` | `51.64` | `436.20` | `168,017,920` | `36,700,160` | `0.22x` | Exact greedy agreement |
| `4096` | `38.95` | `655.83` | `335,790,080` | `62,914,560` | `0.19x` | Exact greedy agreement |
| `8188` | - | - | - | - | - | Dense CUDA OOM |

That is the first useful bigger-model / bigger-KV read on this GPU:

- this 16 GB card comfortably supports exact `2048` and `4096` token SmolLM2 360M runs on the CUDA model path
- the current eager DotCache CUDA path still trails dense CUDA badly on decode at those lengths
- KV-memory savings are already strong at higher context, dropping to about `22%` of dense at `2048` and `19%` at `4096`
- the max-practical `8188` prompt OOMed the dense CUDA baseline on this first pass, so the next optimization loop should focus on `2048` and `4096` first, not on pushing context farther immediately
- the most obvious next bottlenecks on CUDA are exact decode kernel cost and prefill-cache ingest cost, not correctness

### Phase 5 model-integration snapshot

The first exact Llama-family integration path is now implemented in [llama.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/llama.py) on top of [model_kv_cache.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/model_kv_cache.py).

Latest tiny-random Llama smoke runs:

| Backend | Prompt Len | Decode Steps | Decode ms/step | Greedy Agreement | Teacher-Forced Max Abs Logit Drift |
|---|---:|---:|---:|---:|---:|
| `cpu_ref` | `6` | `3` | `2.54` | `1.00` | `0.111` |
| `torch_mps` | `6` | `2` | `18.97` | `1.00` | `8.57e-05` |

That is the right honest read of the new path:

- the narrow Phase 5 harness works end to end on both CPU and this M4 MPS path
- the CPU tiny-random smoke run kept full greedy agreement over the tested steps
- the MPS tiny-random smoke run now also keeps full greedy agreement, and the model-level drift on that local harness is very small
- the real-model TinyLlama path is now the more useful truth source for Phase 5 performance work

Latest real TinyLlama MPS checkpoints on this Mac:

| Change | Decode ms/step | Append ms/step | Host-to-Device Bytes/Step | Greedy Agreement | Teacher-Forced Max Abs Logit Drift |
|---|---:|---:|---:|---:|---:|
| Query scaling fix | `2607.06` | `0.43` | `84480` | `1.00` | `1.5708` |
| Batched KV-head decode | `442.27` | `0.22` | `84480` | `1.00` | `1.6860` |
| Persistent resident tail pages | `187.89` | `32.55` | `22528` | `1.00` | `0.0625` |
| Batched persistent tail uploads | `248.31` | `13.91` | `22528` | `1.00` | `0.0625` |
| On-device query/context decode | `203.11` | `13.58` | `22528` | `1.00` | `0.0625` |
| On-device K/V append | `188.64` | `16.76` | `0` | `1.00` | `0.0625` |
| On-device short-prompt prefill ingest | `149.40` | `4.55` | prefill `0`, decode `0` | `1.00` | `0.0625` |

Longer-prompt exact TinyLlama checks after prewarming deferred full pages once through the prepared-page cache:

| Prompt Len | Prefill Ingest ms | Prefill Ingest H2D Bytes | Decode ms/step | Decode H2D Bytes/Step | Append ms/step | Greedy Agreement |
|---|---:|---:|---:|---:|---:|---:|
| `289` | `95.67` | `1,802,240` | `261.63` | `0` | `5.24` | `1.00` |
| `577` | `106.03` | `3,604,480` | `282.12` | `0` | `4.42` | `1.00` |

Latest dense-KV versus exact DotCache TinyLlama comparison from one loaded model:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Dense Final KV Bytes | DotCache Resident Bytes | DotCache/Dense KV Ratio | Greedy Agreement |
|---|---:|---:|---:|---:|---:|---:|
| `10` | `36.03` | `121.57` | `585,728` | `5,767,168` | `9.85x` | `1.00` |
| `73` | `28.70` | `146.22` | `3,424,256` | `5,767,168` | `1.68x` | `1.00` |
| `145` | `36.46` | `172.30` | `6,668,288` | `5,767,168` | `0.86x` | `1.00` |
| `217` | `34.75` | `172.51` | `9,912,320` | `5,767,168` | `0.58x` | `1.00` |
| `289` | `44.30` | `254.40` | `13,156,352` | `7,569,408` | `0.58x` | `1.00` |
| `433` | `39.60` | `267.12` | `19,644,416` | `7,569,408` | `0.39x` | `1.00` |
| `577` | `79.12` | `279.30` | `26,132,480` | `9,371,648` | `0.36x` | `1.00` |
| `865` | `171.99` | `282.56` | `39,108,608` | `11,173,888` | `0.29x` | `1.00` |

That comparison is the clearest current model-level frontier:

- stock dense KV is still much faster on decode for this exact TinyLlama integration
- DotCache is worse than dense on KV bytes for a very short prompt because the resident prepared-tail machinery dominates at tiny sequence lengths
- the KV-memory crossover happens between about `73` and `145` prompt tokens on this setup; by `865` tokens DotCache uses about `3.5x` less KV memory than dense
- the decode-latency gap narrows as prompts grow, but dense still wins at `865` tokens: about `172 ms/step` dense versus `283 ms/step` DotCache
- resident bytes grow in page-sized steps, which is why `145` and `217` share the same DotCache footprint and `289` / `433` share the next plateau
- greedy token agreement stayed exact across the sweep, so this is a real latency-versus-KV-memory tradeoff rather than a correctness failure

Latest long-prompt optimization checkpoint after skipping redundant `PreparedPageMPS` re-prepare work in the torch decode path:

- targeted `865`-token rerun moved DotCache decode from `282.56 ms/step` down to `241.07 ms/step`
- dense on that rerun was `178.78 ms/step`, so DotCache still trails on latency but the gap is materially smaller
- KV-cache memory stayed the same at a `0.29x` DotCache/dense ratio with full greedy agreement

Latest prefill-ingest optimization checkpoint after batching dense prefill CPU transfers per layer across KV heads:

- targeted `865`-token rerun kept DotCache decode in the same range at `237.98 ms/step`
- prefill-cache ingest dropped from `178.87 ms` down to `102.21 ms`
- the KV-memory result was unchanged at a `0.29x` DotCache/dense ratio with full greedy agreement

Largest practical exact-length TinyLlama comparison point on this M4 so far:

- exact `1536`-token prompt with `max_new_tokens=4` completed successfully
- dense decode: `2274.82 ms/step`
- DotCache decode: `4476.23 ms/step`
- dense final KV bytes: `69,341,184`
- DotCache resident bytes: `16,580,608`
- DotCache/dense KV ratio: `0.24x`
- greedy agreement: `1.00`
- probes at exact `1792` and `1920` prompt tokens OOMed the stock dense MPS baseline on this machine, so `1536` is a good large-case reference point here even though the model's theoretical prompt ceiling with `max_new_tokens=4` is `2044`

First higher-context Llama-family checkpoint beyond TinyLlama:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- theoretical context window: `8192`
- exact `2048`-token prompt with `max_new_tokens=4` completed successfully on this M4
- dense decode: `988.02 ms/step`
- DotCache decode: `1127.39 ms/step`
- dense final KV bytes: `168,017,920`
- DotCache resident bytes: `36,700,160`
- DotCache/dense KV ratio: `0.22x`
- greedy agreement: `1.00`
- probes at exact `3072` and `4096` prompt tokens OOMed the stock dense MPS baseline on this machine, so `2048` is the current practical higher-context reference point here

Latest higher-context decode optimization checkpoint on SmolLM2 360M:

- batched exact MPS decode across KV-head groups inside each layer moved the exact `2048` SmolLM2 decode from `1127.39 ms/step` down to `769.92 ms/step`
- on that rerun, dense landed at `1323.56 ms/step`, so DotCache was about `1.72x` faster on decode while still using only `0.22x` the KV bytes
- greedy agreement stayed at `1.00`
- prefill ingest is still expensive at this size, so the next bottleneck on this model is now prefill rather than decode

Latest higher-context prefill optimization checkpoint on SmolLM2 360M:

- skipping approximate-only page sketch/envelope metadata during exact model-path encoding moved the exact `2048` SmolLM2 prefill-cache ingest from `4233.23 ms` down to `3672.38 ms`
- the same rerun also brought DotCache decode down further to `665.86 ms/step`
- dense landed at `881.04 ms/step` on that run, so DotCache stayed ahead on decode while keeping the same `0.22x` KV-memory ratio

Latest higher-context prewarm scheduling checkpoint on SmolLM2 360M:

- capping MPS prepare batches to bounded page counts moved exact `2048` SmolLM2 prefill-cache ingest from `3672.38 ms` down to `2318.24 ms`
- the same rerun brought DotCache decode down to `399.11 ms/step`
- dense landed at `1260.83 ms/step` on that run, so DotCache was about `3.16x` faster on decode while still using only `0.22x` the KV bytes

Latest higher-context exact-length frontier on SmolLM2 360M:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | DotCache/Dense Decode Speedup | Dense Final KV Bytes | DotCache Resident Bytes | DotCache/Dense KV Ratio | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `256` | `41.79` | `494.46` | `0.08x` | `21,217,280` | `13,762,560` | `0.65x` | One-load frontier sweep |
| `512` | `39.99` | `375.00` | `0.11x` | `42,188,800` | `17,039,360` | `0.40x` | One-load frontier sweep |
| `1024` | `41.65` | `399.80` | `0.10x` | `84,131,840` | `23,592,960` | `0.28x` | One-load frontier sweep |
| `1536` | `172.39` | `456.58` | `0.38x` | `126,074,880` | `30,146,560` | `0.24x` | One-load frontier sweep |
| `2048` | `517.41` | `402.70` | `1.28x` | `168,017,920` | `36,700,160` | `0.22x` | Fresh standalone rerun |

That frontier sharpens the higher-context picture:

- By `512` tokens, DotCache is already down to about `40%` of dense KV bytes on SmolLM2.
- By `1536` tokens, DotCache is using about `24%` of dense KV bytes, but it is still slower on decode in the one-load sweep.
- The fresh standalone `2048` rerun is the best max-case information point on this machine: DotCache decode `402.70 ms/step` versus dense `517.41 ms/step`, while keeping the KV footprint at about `22%` of dense.
- The one-load `2048` sweep result was much noisier for both dense and DotCache, so the standalone rerun is the number to trust for the current max practical SmolLM2 checkpoint.

Latest SmolLM2 midrange decode-scheduling checkpoint:

- skipping redundant grouped decode validation and rechunking inside the prepared MPS fast path moved the one-load exact `1024` point from `399.80` down to `343.21 ms/step`
- the same change moved exact `1536` from `456.58` down to `343.96 ms/step`
- on a new one-load `2048` rerun, DotCache decode landed at `449.96 ms/step`, down from the earlier noisy `659.60 ms/step`
- the standalone `2048` refresh still remains the max-case number to trust most on this machine at `402.70 ms/step`
- an exploratory standalone `1792` point completed successfully with DotCache decode at `487.08 ms/step` and the same strong KV-memory reduction, but the dense side was noisy there too, so it is not yet a clean crossover marker

The current Phase 5 read is:

- exact TinyLlama decode on `torch_mps` is now functionally stable, with full greedy agreement on the short benchmark prompt
- batching decode by KV head was the biggest single model-path performance win so far
- keeping tail pages resident on device removed most decode-time upload churn and sharply improved numerical agreement
- batching tail uploads cut append cost from about `32.6 ms/step` to about `13.9 ms/step`, but total decode latency bounced upward on the short benchmark, so that optimization needs more tuning before we call it a clear end-to-end win
- keeping the exact model-path query/context tensors on device pulled wall-clock decode back down near the better persistent-tail runs while preserving the lower append cost from batched tail uploads
- keeping per-step `K`/`V` append on device removes model-path upload traffic entirely; short-run latency still has some benchmark noise, but the zero-upload result is durable and the end-to-end path is back in the high-100ms/step range instead of the 200ms-plus range
- keeping short-prompt prefill KV ingest on device removes the last obvious host handoff in the exact TinyLlama path for prompts that stay inside the live tail; on the current 10-token prompt, both prefill ingest and decode now run with `0` host-to-device bytes
- for prompts longer than one page, prewarming the deferred full pages once through the prepared-page cache is a better operating point than paying that upload on the first decode step: prefill ingest stays near `100 ms` with the vectorized packing path, decode upload drops to `0`, and long-prompt decode time improves substantially
- vectorizing the CPU bit-packing path takes the next big bite out of long-prompt prefill work: with the deferred-prefill path in place, prefill ingest falls from roughly `975/1892 ms` down to about `87/105 ms` for the `289` and `577` token prompts while preserving exact greedy agreement
- skipping decode-time attention-mask growth for unpadded one-token steps, removing per-layer token-index device sync, and keeping greedy logits on device trims a bit more wrapper/model overhead on long prompts: at `577` tokens, full model decode now lands around `282 ms/step` with no change to exact greedy agreement or page-path upload behavior

Latest exact session baseline on the M4 profile:

- command: `.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8`
- preload: `8.24 ms`
- append: `1.99 ms/step`
- exact decode: `20.58 ms/step`
- full session runtime: `22.57 ms/step`
- max abs error vs CPU full attention: `1.24e-05`

Latest heuristic shortlist comparison on the same workload:

| Policy | Active Pages | Decode ms/step | Session ms/step | Max Abs Error |
|---|---:|---:|---:|---:|
| Exact full context | 19.5 avg | 20.58 | 22.57 | `1.24e-05` |
| Sink 256 + Recent 1024 | 5 | 7.70 | 32.65 | `4.43` |
| Sink 256 + Recent 1024 + relevance `top_k=4` + sketch `1` | 9 | 8.55 | 31.75 | `4.26` |
| Sink 256 + Recent 1024 + relevance `top_k=4` + sketch `4` | 9 | 9.29 | 11.36 | `4.35` |
| Sink 256 + Recent 1024 + relevance `top_k=4` + envelope | 9 | 8.39 | 10.37 | `3.19` |
| Sink 256 + Recent 1024 + sketch shortlist `top_k=8` + exact refine `top_k=4` | 9 | 18.92 | 21.32 | `3.92` |
| Sink 256 + Recent 1024 + envelope shortlist `top_k=8` + exact refine `top_k=4` | 9 | 19.96 | 22.02 | `3.19` |
| Sink 256 + Recent 1024 + approximate old pages | 5 | 21.56 | 45.71 | `4.43` |

Targeted envelope sweep frontier around the promising `256/1024` region:

| Frontier Point | Active Pages | Session ms/step | Max Abs Error |
|---|---:|---:|---:|
| Sink `256` + Recent `1024` + Envelope `top_k=2` | 7 | 7.68 | `3.54` |
| Sink `256` + Recent `1024` + Envelope `top_k=4` | 9 | 8.15 | `3.19` |
| Sink `384` + Recent `1024` + Envelope `top_k=4` | 10 | 8.45 | `3.19` |
| Sink `384` + Recent `1280` + Envelope `top_k=6` | 13 | 10.08 | `3.19` |

Current recommendation from that sweep:

- Fastest useful approximate point: sink `256`, recent `1024`, envelope `top_k=2`.
- Best balanced approximate point: sink `256`, recent `1024`, envelope `top_k=4`.
- Larger sink/recent windows did not materially improve max-abs error enough to justify the extra runtime.

Long-context validation of the balanced `256/1024/4` envelope profile:

| Context | Active Pages | Session ms/step | Max Abs Error |
|---|---:|---:|---:|
| `8192` | 9 | 10.74 | `4.57` |
| `16384` | 9 | 10.37 | `4.73` |

That is a useful result too: the fixed window/profile keeps latency almost flat as context grows, but accuracy degrades, so this should be treated as the best current M4 approximate profile for the 4k regime, not a universal setting.

First context-aware envelope auto-profile validation:

| Context | Profile | Session ms/step | Max Abs Error |
|---|---|---:|---:|
| `4096` | auto (`256/1024/4`) | 10.53 | `3.19` |
| `8192` | auto (`256/2048/4`) | 10.06 | `4.59` |
| `16384` | auto (`256/4096/8`) | 13.68 | `4.70` |

That heuristic was a useful experiment, but not a clear upgrade:

- At `8192`, it was only slightly faster than the fixed profile while giving slightly worse error.
- At `16384`, it reduced error a little versus the fixed profile, but it gave back a lot of latency.
- For now, the fixed `256/1024/4` profile remains the simpler and better default approximate path on this repo.

Budgeted long-context tuner summary:

| Context | Fast Pick | Session ms/step | Max Abs Error | Balanced Pick | Session ms/step | Max Abs Error |
|---|---|---:|---:|---|---:|---:|
| `8192` | `256/2048/4` | 9.21 | `4.59` | `256/1024/8` | 10.86 | `4.51` |
| `16384` | `256/1024/2` | 8.49 | `4.35` | `256/1024/2` | 8.49 | `4.35` |

That is the most useful new data point from this pass:

- The long-context frontier is not monotonic in the way the first auto profile assumed.
- A simple fixed `fast` preset and a simple fixed `balanced` preset are reasonable to keep.
- Context-aware tuning still matters, but it should probably be driven by measured budget/frontier data rather than a naive scale-up rule.

## Working Conclusions

- Exact compressed-domain MPS decode is in very good shape.
- Batched decode and batched preparation were the big wins.
- Heuristic pruning paths buy latency, but current quality loss is still too large.
- Multi-vector sketches are better than a single page mean as a first-pass gate, but they still do not preserve full-context quality.
- A simple page-envelope bound is materially better than sketch gating at about the same latency budget.
- A targeted sweep says `sink=256`, `recent=1024`, `top_k=4` is the best current balance for the M4 profile.
- That same fixed `256/1024/4` profile does not hold quality steady at `8192+`, so longer contexts need a context-scaled retune.
- The first context-aware heuristic did not outperform the simpler fixed profile strongly enough to replace it.
- The budgeted long-context tuner is a better way to explore `8k+` tradeoffs than the first auto-scaling heuristic.
- Precomputing runtime sketches during encode removed the preload/append regression from sketch-based experiments.
- A two-stage shortlist with exact refine improves quality a bit, and score reuse helps a little, but it still gives back most of the latency win from sketch gating.
- Adding exact refine on top of the envelope gate does not currently buy enough to justify the cost.

## Milestone Log

These are the important checkpoints so far. Some numbers come from earlier harness versions, so treat trends as directional rather than perfectly apples-to-apples.

| Commit | Change | Key Result | Takeaway |
|---|---|---|---|
| `d409866` | Batched MPS decode across compatible pages | Warm-cache decode dropped to `5.62 ms/step`; no-cache dropped to `30.19 ms/step` | Keeping more of decode on-device was a major win |
| `a911e05` | Batched MPS page preparation uploads | Host-to-device bytes at `4096` dropped to `1.31 MB`; exact decode prep to about `6.08 ms` post-warmup | Upload batching and compact metadata mattered |
| `484b476` | Session-shaped runtime benchmark | Preload `6.27 ms`, append `2.52 ms/step`, decode `23.16 ms/step` | Runtime phases became measurable separately |
| `627833e` | Sink + recent execution windows | `sink=256`, `recent=1024` cut decode to `7.94 ms/step` with max abs error `4.43` | Naive pruning is fast but too lossy |
| `23c53f6` | Relevance-gated page selection | `top_k=4` cut decode to `9.17 ms/step`; max abs error improved only slightly to `4.26` | Simple page-mean relevance helps latency more than quality |
| `260bfa3` | Approximate fallback for pruned pages | Decode climbed to `21.56 ms/step` while max abs error stayed `4.43` | Summary fallback was not worth the cost |
| `24c1801` | Multi-vector page sketches for gating | Sketch size `4` improved relative error a lot versus sketch size `1` and kept decode at `7.83 ms/step` | Stronger key-side sketches are better than page means, but still not good enough |
| `working tree` | Runtime sketch metadata computed during encode | Exact preload fell back to `8.24 ms`; exact session runtime to `22.57 ms/step`; sketch-gated session runtime to `11.36 ms/step` | Precomputing sketches fixed the preload/append regression and made gating cheap again |
| `working tree` | Two-stage sketch shortlist plus exact refine with reused logits | Decode `18.92 ms/step`; session runtime `21.32 ms/step`; max abs error `3.92` | Reusing shortlisted logits helps a bit, but exact refine is still too expensive to replace either exact decode or the cheap sketch gate |
| `working tree` | Page-envelope relevance gate | Decode `8.39 ms/step`; session runtime `10.37 ms/step`; max abs error `3.19` | A cheap page-level min/max envelope is a much stronger shortlist signal than the sketch gate |
| `working tree` | Envelope shortlist plus exact refine | Decode `19.96 ms/step`; session runtime `22.02 ms/step`; max abs error `3.19` | Exact refine did not improve the envelope shortlist enough to be worth keeping on top |
| `working tree` | Targeted envelope sweep around `256/1024` | Frontier points: `top_k=2` at `7.68 ms/step`, `3.54` error; `top_k=4` at `8.15 ms/step`, `3.19` error | The envelope gate should stay simple; `256/1024/top_k=4` is the best current default approximate profile |
| `working tree` | Long-context validation of tuned envelope profile | `8192` at `10.74 ms/step`, `4.57` error; `16384` at `10.37 ms/step`, `4.73` error | Fixed-window latency scales well, but quality does not; long contexts need retuning |
| `working tree` | First context-aware envelope auto profile | `8192` at `10.06 ms/step`, `4.59` error; `16384` at `13.68 ms/step`, `4.70` error | Context-aware scaling is implementable, but this first heuristic is not yet better enough than the fixed profile to promote |
| `working tree` | Budgeted envelope tuner at `8k/16k` | `8192` fast `256/2048/4`; balanced `256/1024/8`. `16384` fast/balanced both `256/1024/2` | Long-context profile choice is better treated as a budgeted search problem than a simple stepwise heuristic |

## Current Path

What we are keeping:

- Exact full-context MPS decode as the high-quality reference runtime.
- Page-envelope relevance gating as the best current approximate shortlist mode.
- `sink=256`, `recent=1024`, `top_k=4` as the best current M4 approximate profile.
- Fixed named presets in [execution_profiles.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/execution_profiles.py): `m4_envelope_fast` and `m4_envelope_balanced`.
- A wrapper entrypoint for that profile in [run_m4_envelope_session.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_m4_envelope_session.sh).
- A faster fixed wrapper in [run_m4_envelope_fast_session.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_m4_envelope_fast_session.sh).
- An experimental context-aware wrapper in [run_m4_envelope_autoscaled_session.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_m4_envelope_autoscaled_session.sh).
- Session-shaped benchmark harness with separate preload, append, and decode phases.
- A budgeted long-context tuner in [bench_decode_envelope_tuner.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_decode_envelope_tuner.py).
- Raw benchmark history checked into the repo.

What is currently experimental:

- Sink/recent pruning.
- Relevance-gated shortlist selection.
- Exact-refine shortlist selection with reused candidate logits.
- Approximate old-page fallback.
- Multi-vector sketch gating.

What looks next-most-promising:

- Keep exact full-context MPS decode as the production-shaped path while shortlist experiments stay opt-in.
- Promote the envelope gate as the default approximate baseline in any higher-level harnesses we add next.
- Retune the envelope profile as context grows past `4096`, but do it with the budgeted tuner rather than a hand-written scale-up rule.
- Only revisit exact refine if it can improve on the envelope gate without dragging decode time back toward the exact path.

## Recording New Runs

Use the recorder helper to append a machine-readable benchmark entry:

```bash
.venv/bin/python scripts/record_benchmark.py \
  --label "session exact 4k" \
  --notes "Post-sketch baseline on M4 profile" \
  --output benchmarks/results/history.jsonl \
  -- \
  .venv/bin/python benchmarks/bench_decode_session.py \
    --backend torch_mps \
    --config configs/dotcache_m4_mps.yaml \
    --contexts 4096 \
    --decode-steps 8
```

For the model-level dense-vs-DotCache frontier on TinyLlama, use:

```bash
.venv/bin/python scripts/record_benchmark.py \
  --label "llama tinyllama mps dense_vs_dotcache" \
  --notes "One-load TinyLlama comparison sweep across prompt lengths" \
  --output benchmarks/results/history.jsonl \
  -- \
  .venv/bin/python benchmarks/bench_llama_compare.py \
    --backend torch_mps \
    --device mps \
    --max-new-tokens 4 \
    --repeat-counts 1 32 64
```

For the higher-context exact-length frontier on SmolLM2 360M, use:

```bash
.venv/bin/python scripts/record_benchmark.py \
  --label "llama smollm2_360m mps frontier_exact_lengths" \
  --notes "One-load exact-length SmolLM2 frontier sweep on this M4" \
  --output benchmarks/results/history.jsonl \
  -- \
  bash scripts/run_smollm2_frontier_compare.sh
```

For the max practical SmolLM2 point, refresh `2048` on its own so the result is not distorted by the earlier ladder cases sharing the same process:

```bash
.venv/bin/python scripts/record_benchmark.py \
  --label "llama smollm2_360m mps exact_2048_refresh" \
  --notes "Fresh standalone exact-length 2048 rerun on this M4" \
  --output benchmarks/results/history.jsonl \
  -- \
  .venv/bin/python benchmarks/bench_llama_compare.py \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
    --backend torch_mps \
    --device mps \
    --max-new-tokens 4 \
    --repeat-counts \
    --target-prompt-lengths 2048 \
    --continue-on-error
```

If we want, we can later add a small exporter that turns `history.jsonl` into CSV for spreadsheets.
