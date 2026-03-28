# Performance Journal

This file is the high-signal running summary of what we tried, what moved, and what did not.

Raw append-only run history lives in [benchmarks/results/history.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/history.jsonl).
The latest targeted envelope sweep lives in [envelope_sweep_4k.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/envelope_sweep_4k.jsonl).
The latest long-context tuner output lives in [envelope_tuner_8k_16k.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/envelope_tuner_8k_16k.jsonl).

## Current Status

Current branch head: `codex/turbo3-profile-breakdown`

### Teacher-forced quality snapshot

The new [bench_llama_loss.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_loss.py) harness is now the best local quality check for `M1` and `M2`, because greedy agreement and raw max-logit drift were proving too coarse on short continuations.

Latest teacher-forced loss checks on this M4:

| Model | Prefix / Eval | Mode | DotCache Loss Delta | Perplexity Ratio | Token Agreement | Max Abs Logit Drift | DotCache Decode ms/step |
|---|---:|---|---:|---:|---:|---:|---:|
| TinyLlama | `288 / 32` | `K=M0, V=M0` | `-0.00101` | `0.99899` | `1.00` | `8.15` | `3966.95` |
| TinyLlama | `288 / 32` | `K=M0, V=M1` | `-0.00014` | `0.99986` | `1.00` | `12.89` | `2384.70` |
| TinyLlama | `288 / 32` | `K=M2, V=M0` | `+0.00002` | `1.00002` | `1.00` | `12.12` | `2593.00` |
| TinyLlama | `288 / 32` | `K=M2 adaptive, V=M0` | `-0.00135` | `0.99865` | `1.00` | `8.70` | `3230.76` |
| SmolLM2 360M | `1024 / 16` | `K=M0, V=M0` | `-0.00198` | `0.99803` | `1.00` | `7.57` | `2309.06` |
| SmolLM2 360M | `1024 / 16` | `K=M0, V=M1` | `+0.01717` | `1.01732` | `1.00` | `14.54` | `2601.01` |
| SmolLM2 360M | `1024 / 16` | `K=M2, V=M0` | `+0.02865` | `1.02907` | `1.00` | `17.64` | `2512.91` |
| SmolLM2 360M | `1024 / 16` | `K=M2 adaptive, V=M0` | `+0.03948` | `1.04027` | `1.00` | `16.33` | `5158.50` |

What that changed in our read:

- TinyLlama is much more forgiving than the earlier max-logit drift numbers suggested. Both `V`-only `M1` and adaptive `K`-only `M2` kept teacher-forced loss essentially flat on the tested `32`-token continuation.
- SmolLM2 is not nearly as forgiving. At `1024` prefix tokens, both approximate modes materially worsen teacher-forced loss even though teacher-forced token agreement still stays at `1.00`.
- Adaptive segmented `M2` no longer crashes after segment-shape bucketing, but it still loses badly on SmolLM2 and is slower than the fixed segmented variants because mixed segment counts reduce grouped decode efficiency.
- `V`-only `M1` remains the better asymmetric approximate mode than `K`-only `M2` on the tested SmolLM2 slice, but neither is strong enough to replace exact `M0` on the main model path.
- Greedy agreement by itself is not a sufficient quality gate for these modes. Teacher-forced loss/perplexity is now the local quality metric to trust first.

### CUDA handoff from local policy work

The latest local policy work produced a cleaner CUDA handoff package rather than another round of hand-tuned MPS thresholds.

The practical local takeaways to carry forward are:

- `M0 3b` is now a real intermediate tier, and `K=4b, V=3b` is the most plausible first CUDA probe.
- `M3 int8` now works end to end, and the planner can emit recent sealed pages as `M3:int8`.
- TinyLlama has one clearly useful local adaptive profile: conservative middle-layer keys plus aggressive values.
- SmolLM2 still does not have a clearly good adaptive profile, but it does have a safer starting point: early strict keys, deepest keys clamped back to strict, and balanced values.

The concrete handoff artifacts are:

- [tinyllama_cuda_start.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_cuda_start.yaml)
- [smollm2_360m_cuda_start.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_cuda_start.yaml)
- [cuda_next_steps.md](/Users/deanocalver/Documents/Projects/DotCache/docs/cuda_next_steps.md)

### Turbo3 local lane on MPS

Turbo3 now has its own repeatable local runner on this Mac through [run_turbo3_mps_suite.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_turbo3_mps_suite.sh). The current implementation also shares device-resident Turbo3 centroids across prepared pages, and it now has a correct vectorized `3`-bit spill-unpack path in [torch_mps.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/backends/torch_mps.py) that uses advanced indexing instead of repeated-index `torch.gather(...)`.

That materially improved the raw Turbo3 baseline, but the overall model-path answer is still negative:

| Model | Prompt / Eval | Dense Decode ms/step | Turbo3 Decode ms/step | KV Ratio | Agreement | Loss Delta | Perplexity Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| TinyLlama | short compare (`10`) | `396.83` | `1757.78` | `9.85x` | `1.00` | n/a | n/a |
| TinyLlama | exact compare (`289`) | `336.26` | `2631.47` | `0.55x` | `0.25` | n/a | n/a |
| TinyLlama | loss (`288 / 32`) | `450.86` | `3351.38` | n/a | `0.3125` | `+3.16766` | `23.75` |
| SmolLM2 360M | short compare (`7`) | `667.43` | `5711.28` | `12.80x` | `1.00` | n/a | n/a |
| SmolLM2 360M | exact compare (`1024`) | `397.61` | `4306.74` | `0.27x` | `0.25` | n/a | n/a |
| SmolLM2 360M | loss (`1024 / 16`) | `544.12` | `5496.32` | n/a | `0.50` | `+2.45040` | `11.59` |

The most useful apples-to-apples local reference point is now this exact-prompt comparison:

| Model | Prompt | Mode | Decode ms/step | Resident KV Bytes | Agreement | Max Abs Logit Error |
|---|---:|---|---:|---:|---:|---:|
| TinyLlama | `289` | `K=M0, V=M0` | `5730.80` | `7,929,856` | `1.00` | `0.5781` |
| TinyLlama | `289` | `K=M0, V=M1` | `4132.39` | `7,580,672` | `1.00` | `3.2510` |
| TinyLlama | `289` | `K=T3, V=T3` | `2631.47` | `7,208,960` | `0.25` | `26.6152` |
| SmolLM2 360M | `1024` | `K=M0, V=M0` | `5521.76` | `29,360,128` | `1.00` | `0.9769` |
| SmolLM2 360M | `1024` | `K=M0, V=M1` | `6224.08` | `26,820,608` | `1.00` | `3.8877` |
| SmolLM2 360M | `1024` | `K=T3, V=T3` | `4306.74` | `23,068,672` | `0.25` | `15.9824` |

What that means:

- Turbo3 is materially better optimized on MPS than it was before the vectorized spill-unpack work.
- It is still not realistically competitive on the real model path: it buys some KV-memory reduction and, on these noisy exact reruns, better raw decode time than the current-session `M0` / `V`-only `M1` baselines, but only by paying much worse model quality.
- The most useful implementation lesson is concrete: on MPS, repeated-index `gather` is the wrong primitive for this style of low-bit unpack, while advanced indexing is both correct and faster.
- Turbo3 therefore remains a useful reference baseline and implementation-study lane, not a mode we should promote on this Mac.

### One-load local mode profile breakdown

The new [bench_llama_mode_profile.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_mode_profile.py) runner keeps one model loaded and then reconfigures the local DotCache adapter across modes. That gives us a cleaner codec-shape read than the older standalone compare runs, because model load noise is amortized and the backend trace now reports grouped `prepare`, `score`, `softmax`, `mix`, sampled `unpack`, and sampled `fwht`.

TinyLlama exact `289` prompt, one loaded model:

| Mode | Decode ms/step | Score | Mix | Softmax | Unpack | FWHT | Prefill Ingest ms | Resident KV Bytes | Agreement | Max Abs Logit Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` | `5203.45` | `1503.95` | `1409.28` | `100.06` | `2196.55` | `0.00` | `900.88` | `7,929,856` | `1.00` | `0.5781` |
| `Turbo3` | `3917.59` | `882.65` | `687.37` | `105.42` | `802.08` | `30.89` | `8363.96` | `7,208,960` | `0.25` | `26.6152` |

SmolLM2 `1024` prompt, one loaded model:

| Mode | Decode ms/step | Score | Mix | Softmax | Unpack | FWHT | Prefill Ingest ms | Resident KV Bytes | Agreement | Max Abs Logit Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` | `9695.87` | `2686.13` | `2961.70` | `165.62` | `0.00` | `0.00` | `2271.62` | `113,065,984` | `1.00` | `0.9769` |
| `V-only M1` | `5409.58` | `1153.67` | `1163.36` | `153.08` | `648.66` | `0.00` | `10191.47` | `93,405,184` | `1.00` | `2.3926` |
| `Turbo3` | `8156.85` | `2711.92` | `2897.00` | `113.40` | `968.75` | `42.12` | `60617.45` | `39,845,888` | `0.25` | `15.9824` |

What that changed in our read:

- On TinyLlama `289`, Turbo3 is genuinely lighter than `M0` in the hot decode arithmetic. Its grouped `score`, grouped `mix`, and sampled `unpack` totals are all materially smaller, and FWHT overhead is tiny compared with those savings.
- That does not make Turbo3 a viable local mode. The quality collapse is still severe, and its prefill ingest cost is dramatically worse than `M0` on the same one-load profile.
- On SmolLM2 `1024`, `V`-only `M1` is the more interesting DotCache-side trade than Turbo3. It roughly halves grouped `score` and `mix` versus exact `M0`, reduces resident KV bytes, and stays greedy-stable on the short decode, but its write/prefill path is still too expensive.
- On SmolLM2 `1024`, Turbo3 is not even a raw systems win in this one-load profile. Its grouped `score` and `mix` stayed close to `M0`, sampled `unpack` is still substantial, and prefill ingest exploded to more than `60` seconds.
- The most useful concrete lesson so far is still implementation-level: Turbo3 showed that low-bit unpack on MPS needs careful primitive choice, while `V`-only `M1` showed that lighter grouped decode arithmetic does not matter much if prefill ingest balloons.

### Phase 6 vLLM adapter status

The first correctness-first Phase 6 surface is now implemented in [dotcache/integrations/vllm_adapter](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/vllm_adapter).

What landed:

- a pinned `vLLM 0.18.x` compatibility guard
- a block-indexed page bridge keyed by `(layer, kv_head, block_id, kind)`
- finalized-block plus live-partial-block handling on top of the existing exact DotCache runtime
- `dense`, `dotcache_shadow`, and `dotcache_active` adapter modes for Llama-family attention modules
- an offline benchmark entrypoint in [bench_vllm_offline.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_vllm_offline.py)

What we verified locally:

- block ownership is stable
- finalized-block decode matches the quantized-page oracle
- live partial blocks stay visible through the intended `M3` tail path
- active-mode DotCache outputs match shadow-mode DotCache outputs on the tested tiny fake-vLLM Llama path

What is still pending:

- the first real CUDA/vLLM offline benchmark run on the cloud instance
- dense vs shadow vs active timing numbers on real checkpoints
- real shadow-mode drift metrics on vLLM itself

### First NVIDIA CUDA snapshot

The new eager `torch_cuda` backend now runs on an NVIDIA RTX 2000 Ada machine through the same shared torch accelerator path as `torch_mps`.

First decode-step ladder on `torch_cuda`:

| Context | Prepare ms | Decode ms | Host-to-Device Bytes | Max Abs Logit Error | Max Abs Output Error |
|---|---:|---:|---:|---:|---:|
| `64` | `7.22` | `2.31` | `10,240` | `8.58e-06` | `7.15e-07` |
| `256` | `0.35` | `3.10` | `40,960` | `1.24e-05` | `4.05e-06` |
| `1024` | `0.66` | `6.86` | `163,840` | `1.10e-05` | `2.62e-06` |
| `4096` | `2.38` | `21.50` | `655,360` | `1.53e-05` | `7.15e-07` |

First exact session-shaped CUDA checkpoint at `4096` context:

| Backend | Preload ms | Append ms/step | Decode ms/step | Session ms/step | Decode H2D Bytes/Step | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|
| `torch_cuda` | `10.47` | `0.39` | `33.11` | `33.50` | `0` | `2.15e-06` |

Latest tiny-random LLaMA smoke run on CUDA:

| Backend | Prompt Len | Decode Steps | Decode ms/step | Dense Decode ms/step | Greedy Agreement | Teacher-Forced Max Abs Logit Drift |
|---|---:|---:|---:|---:|---:|---:|
| `torch_cuda` | `6` | `3` | `17.65` | `4.24` | `1.00` | `9.79e-05` |

First real TinyLlama dense-vs-DotCache CUDA comparison:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Dense Final KV Bytes | DotCache Resident Bytes | DotCache/Dense KV Ratio | Greedy Agreement |
|---|---:|---:|---:|---:|---:|---:|
| `10` | `40.17` | `72.61` | `585,728` | `5,767,168` | `9.85x` | `1.00` |
| `289` | `22.31` | `132.15` | `13,156,352` | `7,569,408` | `0.58x` | `1.00` |
| `577` | `23.49` | `134.48` | `26,132,480` | `9,371,648` | `0.36x` | `1.00` |

First higher-context CUDA frontier probe on SmolLM2 360M:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Dense Final KV Bytes | DotCache Resident Bytes | DotCache/Dense KV Ratio | Status |
|---|---:|---:|---:|---:|---:|---|
| `2048` | `51.64` | `436.20` | `168,017,920` | `36,700,160` | `0.22x` | Exact greedy agreement |
| `4096` | `38.95` | `655.83` | `335,790,080` | `62,914,560` | `0.19x` | Exact greedy agreement |
| `8188` | - | - | - | - | - | Dense CUDA OOM |

That is the right first CUDA read:

- the shared torch accelerator refactor is numerically stable on CUDA
- the real LLaMA harness runs end to end on CUDA with exact greedy agreement on the recorded cases
- exact compressed-domain decode runs with `0` execution-time host-to-device bytes once pages are prepared
- KV-memory savings show up on real models, but the current eager CUDA path is still a correctness-first parity port, not yet a decode-speed win

Latest exact-length CUDA profiling checkpoint on SmolLM2 360M:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | DotCache Append Runtime ms/step | Prefill Ingest ms | Prefill Ingest H2D | DotCache/Dense KV Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `45.57` | `489.52` | `420.63` | `17.68` | `1118.28` | `25.0 MiB` | `0.26x` |
| `4096` | `52.50` | `735.94` | `675.73` | `17.13` | `2262.02` | `50.0 MiB` | `0.22x` |

That profiling pass clarified the next CUDA phase:

- The hot path is decisively inside `decode_layer_torch`, not append. At both `2048` and `4096`, decode-runtime time dominates DotCache step time.
- Layer `0` is the heaviest per-call decode site on both traces, while the rest of the layers are comparatively flat. On this exact run, most nonzero layers sat around `12.2 ms` per call at `2048` and `20.4 ms` per call at `4096`.
- The remaining model-side overhead outside the explicit DotCache attention timers is material but secondary: about `108.67 ms` total over the `2048` run and `84.65 ms` total over the `4096` run.
- The new CUDA profiler hooks now report per-layer QKV, append, decode, and output-projection timings, plus CUDA allocation and peak-memory counters, through the LLaMA harness and benchmark CLIs.
- A follow-up optimization generalized the prepared-chunk cache to CUDA and cached stacked affine metadata there as well, but that did not materially change the exact `2048` and `4096` decode results. The next useful CUDA optimization will need to change the decode kernel structure itself rather than just remove small stacking overheads.

Latest exact-length CUDA profiling checkpoint on the RTX 5090 32 GB:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | DotCache Append Runtime ms/step | Prefill Ingest ms | DotCache/Dense KV Ratio | DotCache/Dense Total Resident Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `56.35` | `329.23` | `281.53` | `11.88` | `546.31` | `0.22x` | `0.42x` |
| `4096` | `19.54` | `416.66` | `383.47` | `7.97` | `807.13` | `0.19x` | `0.39x` |

The synthetic grouped-64 CUDA microbenchmark on the same 5090 run is directionally consistent with that improvement:

| Shape | Score Speedup | Mix Speedup | Combined Speedup | Max Abs Error |
|---|---:|---:|---:|---:|
| Two-group `64`-dim exact M0 path | `2.68x` | `1.82x` | `2.00x` | `7.63e-05` logits, `1.91e-06` output |

That newer checkpoint changes the CUDA read in a useful way:

- The stronger 5090 hardware plus the fused two-group `64`-dim path materially improve exact SmolLM2 decode latency. Relative to the earlier profiled checkpoint, DotCache decode ms/step dropped from `489.52` to `329.23` at `2048` and from `735.94` to `416.66` at `4096`.
- The breakdown is still clean: unpack time is `0`, softmax is negligible, and score and mix dominate almost evenly. On the trimmed-cache `2048` profile they were `391.53 ms` and `380.79 ms` total; on the `4096` profile they were `549.21 ms` and `529.81 ms` total.
- Append is no longer the interesting cost center on this path. It stayed at `11.88 ms/step` at `2048` and `7.97 ms/step` at `4096`, while decode runtime remained hundreds of milliseconds per step.
- Prefill ingest also got materially faster on the 5090: `546.31 ms` at `2048` and `807.13 ms` at `4096`, versus `1118.28 ms` and `2262.02 ms` on the earlier RTX 2000 Ada profiling run.
- Separating KV residency from prepared-cache overhead restores the original memory picture for exact M0 itself: the DotCache KV ratio is back at `0.22x` for `2048` and `0.19x` for `4096`.
- Trimming the fused prepared chunk representation and avoiding a second grouped cached copy materially improves the total resident footprint. The prepared chunk cache now adds `34.60 MiB` at `2048` and `67.04 MiB` at `4096`, which brings the total resident ratio down to `0.42x` and `0.39x` respectively while keeping the fused decode path active.

Latest exact-length CUDA profiling checkpoint on the RTX 5090 32 GB with an adaptive prepared-chunk budget:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | DotCache Append Runtime ms/step | Prepared Chunk Budget | Prepared Chunk Resident | DotCache/Dense Total Resident Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2048` | `53.49` | `350.59` | `302.67` | `10.59` | `17.50 MiB` | `17.02 MiB` | `0.32x` |
| `4096` | `22.98` | `512.20` | `476.56` | `8.93` | `30.00 MiB` | `29.91 MiB` | `0.28x` |

That adaptive-budget checkpoint is useful, but it changes the tradeoff:

- The lazy synchronized budget logic fixes the earlier pathological regression from recomputing the cache limit on every per-layer append. The recorded `adaptive_chunk_budget_v2` run is the real behavior of the memory-first path.
- At `2048`, the `0.5x`-KV chunk budget is attractive. Total resident ratio drops from `0.42x` to `0.32x`, prepared chunk residency falls from `34.60 MiB` to `17.02 MiB`, and decode still lands in the same rough band as the trimmed-cache checkpoint.
- At `4096`, the same policy is much more clearly a memory-first trade. Total resident ratio drops further from `0.39x` to `0.28x`, but decode slows from `416.66 ms/step` to `512.20 ms/step`.
- The right read is that this budgeted path is not a free win. It is a useful guardrail for memory-sensitive deployments, but the 5090 data says the uncapped trimmed fused cache remains the better throughput-oriented default for longer exact contexts unless we make the cache policy more workload-shaped.

### Recent CUDA negative results on the RTX 5090 32 GB

We ran three more CUDA experiments against the real `meta-llama/Llama-3.2-3B-Instruct` exact-length `4096` lane after the FP32 affine-metadata win. All three kept `greedy_token_agreement_rate = 1.0`, but none beat the current stable checkpoint, so all three were reverted.

Current stable checkpoint after those reversions:

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | KV Resident Bytes | Total Resident Bytes | Prepared Chunk Resident |
|---|---:|---:|---:|---:|---:|---:|
| `4096` | `46.72` | `617.02` | `564.93` | `205,520,896` | `271,056,896` | `65,536,000` |

Failed follow-up experiments:

| Experiment | Profiled Decode ms/step | Score | Mix | Non-profiled Decode ms/step | Total Resident Bytes | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Static-pages plus dense-tail specialized merge | `651.88` | `255.43` | `243.20` | `614.34` | `271,056,896` | Correct but not faster |
| Fused `4 x 32 -> 128` PyTorch matmul path | `653.23` | `272.40` | `262.99` | `707.77` | `272,433,152` | Regressed badly |
| Triton grouped `4 x 32 -> 128` fused score/mix path | `638.32` | `264.25` | `256.60` | `631.86` | `251,658,240` | Memory win only, still slower |

What that changed in our read:

- The remaining `4096` Llama 3.2 bottleneck is not chunk-boundary bookkeeping. The specialized static-plus-tail merge removed some structure, but it did not move the real end-to-end number enough to justify extra complexity.
- Wider fused matmuls over precomputed `codes * scales` are not automatically a win on this workload. The eager `4 x 32 -> 128` PyTorch fusion regressed sharply even though the narrower `2 x 64` fused path remains useful on the models that match it.
- The first Triton grouped `128`-dim path was directionally better than the eager fused-PyTorch attempt and cut prepared-chunk residency from `65.54 MiB` to `46.14 MiB`, but it still lost to the stable checkpoint on decode time.
- The next useful kernel step therefore needs to drop below the current prepared-chunk abstraction. If we want a real CUDA/Triton win on the Llama 3.2 `128`-dim lane, the kernel must consume packed page words directly and fuse unpack plus affine score or mix, instead of first materializing the current unpacked grouped tensors or fused scaled-code buffers.

### Latest Llama 3.2 packed-word CUDA checkpoint on the RTX 5090 32 GB

That next lower-level idea did pay off once we kept it narrow. For the CUDA-only `M0` grouped `head_dim=128` lane, the grouped prepared-chunk cache now retains packed page payload words plus affine scales and bias, and the hot score/mix path unpacks one packed word at a time instead of caching full unpacked `[... , 32]` code tensors.

Latest exact `4096` checkpoint on `meta-llama/Llama-3.2-3B-Instruct`:

| Run | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | Score | Mix | Unpack | KV Resident Bytes | Total Resident Bytes | Prepared Chunk Resident | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous stable checkpoint | `46.72` | `617.02` | `564.93` | n/a | n/a | n/a | `205,520,896` | `271,056,896` | `65,536,000` | `1.0` |
| New profiled packed-word run | `108.77` | `569.00` | `523.33` | `221.63` | `207.63` | `48.90` | `205,520,896` | `271,581,184` | `66,060,288` | `1.0` |
| New non-profiled packed-word run | `95.63` | `575.75` | `521.39` | n/a | n/a | n/a | `205,520,896` | `271,581,184` | `66,060,288` | `1.0` |

What that changed in our read:

- The grouped `128`-dim CUDA lane was in fact paying too much for cached unpacked codes. Replacing that cache with packed payload words plus affine metadata improved the real exact `4096` Llama 3.2 decode checkpoint from `617.02 ms/step` to `575.75 ms/step`.
- The profile breakdown is healthier than the earlier eager or Triton grouped `128`-dim experiments. Score fell into the `221.63 ms` band, mix into `207.63 ms`, and the new explicit unpack work stayed bounded at `48.90 ms`.
- The memory trade is basically flat. KV residency stayed unchanged, total resident bytes moved only slightly from `271.06 MiB` to `271.58 MiB`, and prepared-chunk residency stayed in the same rough `~66 MiB` band.
- The important systems detail is that the packed-word path keeps peak temporary decode materialization small. On the recorded profile the trace reported only `1.0 MiB` max temporary bytes while the grouped path executed `896` unpack calls across the full decode step.
- This is the first lower-level grouped `128`-dim CUDA change that beat the prior stable checkpoint and kept exact greedy agreement, so it should remain the default path for this exact lane.

### Latest Llama 3.2 unpack-once CUDA checkpoint on the RTX 5090 32 GB

There was still avoidable overhead inside that packed-word path. The next refinement kept the same CUDA-only grouped `128`-dim cache shape, but changed score and mix to unpack each `32`-symbol group once per call and then reuse the unpacked group for the existing grouped `32`-dim matmul, instead of doing four separate unpack-plus-word-matmul passes.

Latest exact `4096` checkpoint on `meta-llama/Llama-3.2-3B-Instruct` after that refinement:

| Run | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | Score | Mix | Unpack | Unpack Calls | Max Temporary Bytes | KV Resident Bytes | Total Resident Bytes | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous packed-word checkpoint | `95.63` | `575.75` | `521.39` | `221.63` | `207.63` | `48.90` | `896` | `1.0 MiB` | `205,520,896` | `271,581,184` | `1.0` |
| New profiled unpack-once run | `99.63` | `533.48` | `456.93` | `186.32` | `175.64` | `15.20` | `224` | `4.0 MiB` | `205,520,896` | `271,581,184` | `1.0` |
| New non-profiled unpack-once run | `100.32` | `513.68` | `467.18` | n/a | n/a | n/a | n/a | n/a | `205,520,896` | `271,581,184` | `1.0` |

What that changed in our read:

- The remaining unpack overhead was mostly call structure, not unavoidable arithmetic. Collapsing four word-level unpack passes into one per grouped `32`-symbol chunk cut profiled unpack time from `48.90 ms` to `15.20 ms`.
- The score and mix bands also improved materially once the grouped path stopped bouncing through the smaller per-word loop. On the recorded profile, score fell from `221.63 ms` to `186.32 ms` and mix from `207.63 ms` to `175.64 ms`.
- The real end-to-end checkpoint moved again in the right direction: non-profiled exact `4096` decode improved from `575.75 ms/step` to `513.68 ms/step` while keeping exact greedy agreement.
- The tradeoff is a larger temporary decode buffer. Peak temporary bytes rose from `1.0 MiB` to `4.0 MiB`, but that is still small relative to the overall resident footprint and the total resident bytes stayed flat.
- This is a cleaner steady-state shape for the current eager CUDA path. The next worthwhile kernel step is now below this unpack-once grouped path, not another reorganization of the same eager tensor work.

Rejected follow-up Triton score-only kernel on the same lane:

| Experiment | Profiled Decode ms/step | Decode Runtime ms/step | Score | Mix | Unpack | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Triton page-native score-only kernel over packed grouped `32`-symbol pages | `940.64` | `889.81` | `608.91` | `183.77` | `9.53` | Reverted |

What that changed in our read:

- The first page-native Triton direction did reduce the explicit unpack band, but it made the score path dramatically worse on the real `4096` Llama 3.2 workload.
- The failure mode is useful: dropping below the eager path is not enough by itself. A kernel that wins must preserve the score-side memory access and reduction structure of the workload, not just fuse the arithmetic.
- The current best checkpoint therefore remains the eager unpack-once grouped path above, and any next kernel pass should target a fuller fused score-plus-mix design or a substantially better score tiling strategy rather than this score-only Triton shape.

Rejected follow-up CUDA grouped output-only decode swap:

| Experiment | Profiled Decode ms/step | Decode Runtime ms/step | Score | Mix | Softmax | Unpack | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Route CUDA grouped decode through the existing output-only streaming score-softmax-mix path | `575.08` | `519.68` | `223.98` | `185.22` | `6.96` | `14.71` | Reverted |

What that changed in our read:

- The broader algorithmic fusion was not free. Avoiding full logits materialization on CUDA increased the softmax call count and pushed enough extra work into the streaming normalize path to erase the gain.
- On this exact lane, the current unpack-once eager grouped decode remains better than the output-only streaming variant. The eager path keeps softmax essentially negligible, while the streaming route moved it into a measurable new cost center.
- That means the next useful step still has to be lower-level than control-flow reshaping. We need a truly fused kernel or a better score/mix implementation, not just a different orchestration of the same eager operators.

### Latest Llama 3.2 pairwise grouped-batch CUDA checkpoint on the RTX 5090 32 GB

The next useful CUDA step turned out not to be a full four-group batch. Batching all four `32`-dim groups together cut unpack calls too aggressively and drove temporary decode materialization up to `16 MiB`, which regressed the profiled exact `4096` lane to `582.28 ms/step` even though the explicit unpack band fell to `5.65 ms`. The keepable version is narrower: batch the packed grouped `128`-dim path in pairwise `2`-group slices so score and mix launch fewer small kernels without paying the full temporary-memory penalty of the all-four-group version.

Latest exact `4096` checkpoint on `meta-llama/Llama-3.2-3B-Instruct` after that refinement:

| Run | Dense Decode ms/step | DotCache Decode ms/step | DotCache Decode Runtime ms/step | Score | Mix | Unpack | Unpack Calls | Max Temporary Bytes | KV Resident Bytes | Total Resident Bytes | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous unpack-once checkpoint | `100.32` | `513.68` | `467.18` | `186.32` | `175.64` | `15.20` | `224` | `4.0 MiB` | `205,520,896` | `271,581,184` | `1.0` |
| New profiled pairwise grouped-batch run | `103.61` | `511.98` | `467.09` | `193.37` | `176.75` | `7.05` | `112` | `8.0 MiB` | `205,520,896` | `271,581,184` | `1.0` |
| New non-profiled pairwise grouped-batch run | `100.60` | `498.32` | `452.72` | n/a | n/a | n/a | n/a | n/a | `205,520,896` | `271,581,184` | `1.0` |

What that changed in our read:

- The best next CUDA win was a launch-shape adjustment inside the existing eager grouped `32`-dim math, not a wider `128`-dim fusion. Pairwise batching keeps the arithmetic on the favorable `32`-wide kernels while cutting packed-path unpack calls from `224` to `112`.
- The explicit unpack band dropped materially again, from `15.20 ms` to `7.05 ms`, without changing the resident memory footprint or losing exact greedy agreement on the real `4096` Llama 3.2 lane.
- The full four-group version overshot the temporary-memory budget and lost despite even lower unpack time. The pairwise version is the better balance: peak temporary bytes rise from `4.0 MiB` to `8.0 MiB`, which is still small enough to keep the end-to-end checkpoint moving in the right direction.
- The new steady-state checkpoint is the best one on this lane so far. Non-profiled exact `4096` decode improved from `513.68 ms/step` to `498.32 ms/step`, while the profiled run moved from `533.48 ms/step` to `511.98 ms/step`.

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

Latest SmolLM2 FP32-affine metadata checkpoint:

- storing M0 affine `scales` and `bias` as FP32 on device at page-preparation time removed the repeated decode-time casts from the hot MPS kernels
- on the one-load exact-length rerun, DotCache decode moved to `306.09 ms/step` at `1024`, `262.43 ms/step` at `1536`, and `402.62 ms/step` at `2048`
- the matching DotCache resident KV bytes increased from about `23.59/30.15/36.70 MB` to about `26.21/34.08/41.94 MB` at `1024/1536/2048`
- a fresh standalone `1536` rerun landed at `285.99 ms/step` DotCache decode with the same greedy agreement and a `34.08 MB` resident KV footprint
- the honest read is that this is a good MPS trade on this machine: a noticeable DotCache decode-speed win for a moderate KV-resident-memory increase, while staying far below dense KV bytes at the same prompt lengths

Latest SmolLM2 static prepared-chunk cache checkpoint:

- preparing static prefill pages once in the model decode-view path unlocked reuse of the new stacked M0 chunk cache across repeated grouped MPS decodes instead of rebuilding fresh prepared-page identities every step
- on the one-load exact-length rerun, DotCache decode moved to `252.20 ms/step` at `1024`, `234.30 ms/step` at `1536`, and `578.09 ms/step` at `2048`
- versus the earlier FP32-affine metadata checkpoint, that is a clear DotCache-side improvement at `1024` and `1536`:
  `306.09 -> 252.20 ms/step` at `1024`
  `262.43 -> 234.30 ms/step` at `1536`
- the tradeoff is higher resident DotCache KV bytes because the stacked static chunk tensors now stay resident:
  `26.21 -> 29.36 MB` at `1024`
  `34.08 -> 38.80 MB` at `1536`
  `41.94 -> 48.23 MB` at `2048`
- standalone `1536` and `2048` refreshes were still noisy on the dense side and also showed very unstable prefill timings on MPS, so they should not replace the earlier trusted frontier points by themselves
- the honest read is that this is another good MPS trade for the real model path: better DotCache decode throughput at the cost of a moderate additional resident-memory increase inside DotCache, while still remaining well below dense KV bytes at the same prompt lengths

Rejected SmolLM2 hard-capped prepared-chunk cache checkpoint:

- bounding the static prepared-chunk cache to a hard `4 MiB` resident budget preserved correctness, but it gave back too much of the decode win
- on the one-load exact-length rerun, DotCache decode regressed to `563.95 ms/step` at `1024` and `560.93 ms/step` at `1536`
- resident DotCache KV did come down modestly to about `29.36 MB` at `1024` and `38.21 MB` at `1536`, but that was nowhere near enough to justify the large throughput loss
- this should be treated as a losing case: the hard budget was too tight for the grouped static-page reuse pattern on this model

Latest SmolLM2 payload-only prepared-chunk cache checkpoint:

- shrinking the static prepared-chunk cache to retain stacked payload tensors only, while recomputing the lighter affine stacks on demand, gave a much healthier speed-memory trade than the hard-capped experiment
- on the one-load exact-length rerun, DotCache decode landed at:
  `249.34 ms/step` at `1024`
  `282.24 ms/step` at `1536`
  `266.04 ms/step` at `2048`

### Experimental M1 LUT asymmetry snapshot

The first paper-shaped `M1` LUT experiments are now wired through the real-model harness and recorded in [history.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/history.jsonl), including per-kind page counts and fallback/error stats.

The most useful asymmetric result so far is `V`-only `M1`: keep keys on `M0`, switch values to refined shared-LUT `M1`, and leave the page-level `M1 -> M0` fallback enabled.

| Model | Prompt Len | K Mode | V Mode | Decode ms/step | Resident Bytes | DotCache/Dense KV Ratio | Max Abs Logit Drift | Notes |
|---|---:|---|---|---:|---:|---:|---:|---|
| `TinyLlama 1.1B` | `289` | `M0` | `M1` | `340.89` | `7,580,672` | `0.576x` | `3.2510` | `88` value pages in `M1`, `0` fallbacks |
| `SmolLM2 360M` | `1024` | `M0` | `M1` | `317.52` | `25,772,032` | `0.306x` | `3.8877` | `640` value pages in `M1`, `0` fallbacks |

Compared with the earlier experiments:

- `V`-only `M1` is clearly better than `K`-only `M1`; the logit drift is materially lower while the KV-memory ratio is the same.
- `V`-only `M1` is also better than the earlier full `M1` runs on drift.
- But on these current MPS model-path checks it is still not better than plain `M0` overall, because `M0` keeps much smaller drift and better decode latency.

So the honest current read is:

- the paper’s K/V asymmetry shows up clearly on this machine
- `M1` on values is the only promising local `M1` variant so far
- the page-level fallback hook works, but it is not rescuing quality on these workloads because almost no pages trip it under the current reconstruction-error metric
- the next meaningful `M1` step should be a stronger value-side codec or a better fallback signal, not more threshold guessing

We then tried both of those next steps together as an explicit experiment, not a new default:

- stronger value-side codec: segmented `M1` LUTs for values (`m1_segment_count_v=2`)
- stronger fallback signal: token-wise relative error `p95` in addition to the page-average reconstruction metric

That experiment finally made fallback fire on real prompts:

| Model | Prompt Len | V `M1` Pages | V Fallback Pages | Decode ms/step | Resident Bytes | Max Abs Logit Drift | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `TinyLlama 1.1B` | `289` | `83` | `5` | `374.57` | `7,611,136` | `2.7070` | Drift improved, latency worsened |
| `SmolLM2 360M` | `1024` | `620` | `20` | `378.57` | `25,930,752` | `6.6113` | Both latency and drift worsened |

So the honest read is:

- the improved fallback metric is doing real work now
- but the segmented value-side codec is not a win on these current MPS model-path checks
- the repo keeps both features as explicit tuning knobs, while the safer single-segment `M1` path remains the default

### First M2 key-sketch snapshot

The first real `M2` implementation is now in the repo as a key-only approximate mode: page-local fixed-projection key sketches stored per token/group, with exact `M0` values left unchanged. This is the first codec-backed key-side approximation path, not just a page-summary gating heuristic.

The first real-model `K=M2, V=M0` checks on MPS were not good enough:

| Model | Prompt Len | Sketch Dim | Decode ms/step | Resident Bytes | DotCache/Dense KV Ratio | Greedy Agreement | Max Abs Logit Drift | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `TinyLlama 1.1B` | `289` | `8` | `248.02` | `8,290,304` | `0.630x` | `0.50` | `27.54` | Recorded in `history.jsonl` |
| `SmolLM2 360M` | `1024` | `8` | `521.35` | `30,932,992` | `0.368x` | `0.25` | `12.65` | Recorded in `history.jsonl` |

I also probed a wider sketch for quick quality-vs-cost context:

| Model | Prompt Len | Sketch Dim | Decode ms/step | Resident Bytes | Greedy Agreement | Max Abs Logit Drift | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| `TinyLlama 1.1B` | `289` | `16` | `198.57` | `9,732,096` | `0.50` | `26.09` | Slightly better latency, still very wrong |
| `SmolLM2 360M` | `1024` | `16` | `435.16` | `41,418,752` | `0.75` | `20.46` | Better agreement, worse drift/memory, still not viable |

So the honest read is:

- this first `M2` is useful infrastructure, because it proves a real key-only approximate codec can be wired through the exact runtime and model harness cleanly
- but this fixed random-projection sketch family is not good enough yet on the model path
- compared with the earlier envelope/summary gating experiments, it still does not buy the quality we need for the cost
- the next meaningful `M2` step would need a better key sketch family, not just another width tweak

### Adaptive M2 low-rank snapshot

The next `M2` step replaced the fixed random projections with a page-local low-rank basis: each key page now stores per-token coefficients plus a shared per-group basis, and scoring projects the query through that stored basis instead of through a fixed random matrix.

That adaptive sketch family is materially better than the first `M2` version on the real model path:

| Model | Prompt Len | Rank | Decode ms/step | Resident Bytes | DotCache/Dense KV Ratio | Greedy Agreement | Max Abs Logit Drift | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `TinyLlama 1.1B` | `289` | `8` | `162.25` | `8,470,528` | `0.644x` | `1.00` | `7.45` | Much better than fixed random `M2`, still worse than exact `M0` |
| `SmolLM2 360M` | `1024` | `8` | `274.67` | `32,243,712` | `0.383x` | `1.00` | `12.10` | Major quality recovery, still meaningfully driftier than `M0` |

Compared with the earlier fixed random-projection `M2` at the same sketch width:

- TinyLlama `289`: greedy agreement recovered from `0.50` to `1.00`, max abs drift improved from `27.54` to `7.45`, and decode improved from `248.02` to `162.25 ms/step`
- SmolLM2 `1024`: greedy agreement recovered from `0.25` to `1.00`, max abs drift improved from `12.65` to `12.10`, and decode improved from `521.35` to `274.67 ms/step`

The honest read now is:

- adaptive low-rank `M2` is the first key-only approximate mode that looks technically credible on the real MPS model harness
- it is still not good enough to replace exact `M0` on the model path, because the teacher-forced drift remains high even though greedy agreement recovered on these short checks
- the main remaining quality question is whether a stronger basis construction, a mixed exact/approximate fallback, or a narrower use case such as prefiltering older pages can exploit this much better `M2` signal without paying full exact cost

### Experimental adaptive M2 prefilter snapshot

I also wired the adaptive `M2` basis in as a sidecar on exact `M0` key pages so decode can do approximate page selection first and then exact rescoring/mixing only on the shortlisted pages. This is fully opt-in through `m2_prefilter_top_k`.

The first real-model result is a losing case on this M4:

| Model | Prompt Len | Top-K | Decode ms/step | Resident Bytes | DotCache/Dense KV Ratio | Greedy Agreement | Max Abs Logit Drift | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SmolLM2 360M` | `1024` | `2` | `649.74` | `38,010,880` | `0.452x` | `1.00` | `5.25` | Shortlist works, but the sidecar + prefilter cost outweighs the exact decode saved |

Useful instrumentation from that run:

- `m2_sidecar_pages=640`: all static key pages carried the adaptive `M2` sidecar as intended
- `m2_prefilter_candidate_pages=1920` and `m2_prefilter_selected_pages=1440`: the shortlist is actively dropping static pages, not just passing everything through
- despite that, total decode time still regressed badly versus the exact `M0` path on the same model family

So the honest read is:

- adaptive `M2` is more promising as a page-selection signal than as a full replacement codec
- but this first sidecar-prefilter implementation is not yet a win on the real MPS harness
- the next meaningful improvement would need to make shortlist scoring much cheaper, or apply it only in regimes where page count is high enough to amortize the extra work

Experimental SmolLM2 key-only prepared-chunk cache checkpoint:

- forcing the static prepared-chunk cache to keep only key-side chunks was a reasonable workload-shaped hypothesis, since score-side chunk reuse is often more valuable than value-side reuse
- on the one-load exact-length rerun, DotCache decode landed at:
  `226.81 ms/step` at `1024`
  `276.20 ms/step` at `1536`
  `396.96 ms/step` at `2048`
- that is a small improvement over the payload-only default at `1024` and `1536`, but a clear regression at `2048`
- the recorded resident DotCache KV bytes were effectively unchanged from the payload-only checkpoint on this ladder, so this did not buy the memory reduction we would want in exchange for the longer-context regression
- the honest read is that key-only cache selection is worth keeping as experimental infrastructure, but it should not replace the broader payload-only default on this branch

Rejected M3 FP32 escape-payload experiment:

- keeping `M3` escape payloads resident as FP32 on device looked promising for live-tail decode, but the real-model measurements went the wrong way
- short TinyLlama decode regressed to about `179.62 ms/step` and resident KV jumped to about `11.53 MB`
- SmolLM2 exact `256` also regressed to about `160.61 ms/step` with resident KV rising to about `24.90 MB`, which was worse than the existing mixed-path checkpoint
- we kept the benchmark records in [history.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/history.jsonl) and reverted the code, so this should be treated as an explored losing case rather than the new path forward

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

For the current best asymmetric `V`-only `M1` checks on the real model path, use:

```bash
.venv/bin/python scripts/record_benchmark.py \
  --label "llama smollm2 mps v_only_m1" \
  --notes "Exact SmolLM2 360M comparison on MPS with M0 keys and refined+tanh M1 values only." \
  --output benchmarks/results/history.jsonl \
  -- \
  .venv/bin/python benchmarks/bench_llama_compare.py \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
    --backend torch_mps \
    --device mps \
    --max-new-tokens 4 \
    --repeat-counts \
    --target-prompt-lengths 1024 \
    --default-mode-k M0 \
    --quant-scheme-k affine \
    --default-mode-v M1 \
    --quant-scheme-v lut \
    --lut-refine-steps 6 \
    --preconditioner tanh \
    --m1-fallback-to-m0 \
    --m1-error-threshold 0.2
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

## Qwen2.5 3B Local Read

Qwen2.5 3B is now confirmed runnable on this M4 through the native HF path, but it is clearly a stretch-model lane rather than a comfortable local development target.

- Exact `256` prompt tokens:
  - dense decode `1658.82 ms/step`
  - DotCache decode `7386.23 ms/step`
  - DotCache KV ratio `0.68x`
- Exact `512` prompt tokens:
  - dense decode `1073.71 ms/step`
  - DotCache decode `7156.34 ms/step`
  - DotCache KV ratio `0.44x`
- Exact `1024` prompt tokens:
  - dense decode `74658.40 ms/step`
  - DotCache decode `9637.94 ms/step`
  - DotCache KV ratio `1.71x`

The `1024` dense point is clearly noisy and should not be treated as a clean crossover claim. The trustworthy takeaway is simpler: this Mac can host and run Qwen2.5 3B for smoke tests and frontier checks up to at least exact `1024` prompt tokens, but it is near the limit of what is useful for sustained 3B-class optimization work.

## Local M0 3-bit Probe

`M0 3b` now works on the local MPS path through the general encode/prepare/decode path. It is not yet part of the direct full-prefill torch fast path, so these numbers are best read as planning hints rather than a polished systems result.

- TinyLlama exact `577`, `K=3b, V=3b`
  - dense decode `1438.08 ms/step`
  - DotCache decode `14415.48 ms/step`
  - KV ratio `0.360x`
  - greedy agreement `1.0`
- TinyLlama exact `577`, `K=3b, V=4b`
  - dense decode `653.61 ms/step`
  - DotCache decode `9554.66 ms/step`
  - KV ratio `0.374x`
  - greedy agreement `1.0`
- TinyLlama exact `577`, `K=4b, V=3b`
  - dense decode `731.19 ms/step`
  - DotCache decode `8488.08 ms/step`
  - KV ratio `0.374x`
  - greedy agreement `1.0`

Short teacher-forced TinyLlama checks (`304 / 288 / 4`) also stayed clean:

- `K=3b, V=4b`
  - loss delta `-0.00317`
  - token agreement `1.0`
- `K=4b, V=3b`
  - loss delta `-0.00250`
  - token agreement `1.0`

The local hint is that `3b` looks more plausible as a value-side tier than a key-side one. `K=4b, V=3b` was the best asymmetric TinyLlama run on both runtime and logit drift, while `K=3b, V=4b` was still viable but slightly worse.

One SmolLM2 360M exact `1024` probe on the more promising asymmetric split showed the same direction:

- `K=4b, V=3b`
  - dense decode `878.59 ms/step`
  - DotCache decode `12189.15 ms/step`
  - KV ratio `0.297x`
  - greedy agreement `1.0`

So the honest local conclusion is:

- `M0 3b` is a real new planning tier now
- it appears quality-viable on TinyLlama and plausible on SmolLM2
- the current MPS implementation is still much too slow to promote it as a runtime win
- if we use `3b` in policy work, the best first guess is `K=4b, V=3b`, not `3b` everywhere

I then added a one-load real-model probe in [bench_llama_mixed_bits_profile.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_mixed_bits_profile.py) with the convenience wrapper [run_m0_mixed_bits_mps_probe.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_m0_mixed_bits_mps_probe.sh) so `K=4b, V=4b` and `K=4b, V=3b` could be compared against the same loaded model and the same grouped fused-cache policy.

That gave a more honest systems read:

| Model | Prompt | Split | Dense Decode ms/step | DotCache Decode ms/step | Prefill Ingest ms | Resident Bytes | Agreement | Max Abs Logit Drift |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `TinyLlama 1.1B` | `577` | `K=4b, V=4b` | `1309.16` | `25681.65` | `3837.36` | `14,958,592` | `1.00` | `0.6074` |
| `TinyLlama 1.1B` | `577` | `K=4b, V=3b` | `564.23` | `21779.50` | `12250.44` | `14,598,144` | `1.00` | `1.2207` |
| `SmolLM2 360M` | `1024` | `K=4b, V=4b` | `1415.17` | `36945.79` | `2083.56` | `39,190,528` | `1.00` | `0.9612` |
| `SmolLM2 360M` | `1024` | `K=4b, V=3b` | `446.14` | `37179.83` | `13848.80` | `37,339,136` | `1.00` | `2.1914` |

So the follow-up conclusion is more nuanced than the earlier cold-load probes:

- `K=4b, V=3b` does reduce resident bytes on both models.
- On TinyLlama, it also improved decode time under the one-load comparison.
- On SmolLM2, decode was effectively flat-to-worse while prefill ingest regressed badly.
- The main blocker for `V=3b` on this Mac is now clearly the write/prefill side, not grouped decode.
- That still makes `K=4b, V=3b` a useful CUDA hint, but only if the write path there is materially cheaper than it is on MPS.

I then optimized the host `3b` payload builder itself in [packing.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/packing.py) and [page_format.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/page_format.py):

- the old spill path packed `3b` row-by-row in Python
- the new path packs per symbol across all rows at once
- `group_major` payload building now packs all groups in one vectorized call instead of looping group-by-group

The dedicated write-path microbench in [bench_m0_write_micro.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_m0_write_micro.py) shows that this was a real bottleneck:

- synthetic write microbench (`256 x 128`, `group_size=32`)
  - `3b` quantize: `0.284 ms`
  - `3b` payload build: `0.230 ms`
  - `3b` legacy payload build: `81.236 ms`
  - `3b` payload speedup vs legacy: about `352.6x`
  - `4b` quantize: `0.916 ms`
  - `4b` payload build: `0.373 ms`

And the post-change one-load TinyLlama rerun confirms the win reached the model path:

| Model | Prompt | Split | DotCache Decode ms/step | Prefill Ingest ms | Resident Bytes | Agreement |
|---|---:|---|---:|---:|---:|---:|
| `TinyLlama 1.1B` | `577` | `K=4b, V=4b` | `24736.65` | `4233.71` | `14,958,592` | `1.00` |
| `TinyLlama 1.1B` | `577` | `K=4b, V=3b` | `22635.27` | `2399.05` | `14,598,144` | `1.00` |

Compared with the earlier one-load TinyLlama mixed-bits probe, `K=4b, V=3b` prefill ingest improved from `12250.44 ms` down to `2399.05 ms`, about `5.1x` better. So host payload packing really was a large part of the `V=3b` write-side pain on this Mac.

The remaining honest read is:

- `K=4b, V=3b` is now a much more credible local lane on TinyLlama
- the write-path fix materially improved prefill ingest
- the same write-path win also shows up on SmolLM2
- the next likely bottleneck after payload packing is the remaining encode/quantize work, not grouped decode

The post-fix SmolLM2 one-load rerun says the same thing more clearly:

| Model | Prompt | Split | DotCache Decode ms/step | Prefill Ingest ms | Resident Bytes | Agreement | Max Abs Logit Drift |
|---|---:|---|---:|---:|---:|---:|---:|
| `SmolLM2 360M` | `1024` | `K=4b, V=4b` | `47037.17` | `1802.12` | `39,190,528` | `1.00` | `0.9612` |
| `SmolLM2 360M` | `1024` | `K=4b, V=3b` | `37030.24` | `3203.19` | `37,339,136` | `1.00` | `2.1914` |

Compared with the earlier pre-fix one-load SmolLM2 mixed-bits probe:

- `K=4b, V=3b` prefill ingest improved from `13848.80 ms` to `3203.19 ms`, about `4.3x` better
- `K=4b, V=4b` prefill ingest also improved slightly, from `2083.56 ms` to `1802.12 ms`
- decode did not become a clear win; it only moved from roughly `37179.83` to `37030.24 ms/step`

So the updated cross-model read is:

- the vectorized `3b` payload builder fixed a real write-path bottleneck on both TinyLlama and SmolLM2
- `K=4b, V=3b` is now much more plausible as a systems experiment than it was before
- but on SmolLM2 the remaining question is no longer “can we write it fast enough?” so much as “is the extra quality loss worth the modest KV-memory reduction without a decode win?”

One backend-focused optimization pass did move the local `3b` decode shape in the right direction. Static `M0 3b` pages on MPS now build a fused pre-scaled prepared chunk, not just cached unpacked per-group codes. On a synthetic cached static-page decode microbench (`4` pages, `head_dim=64`, `tokens_per_page=16`):

- without prepared chunk cache: `142.71 ms`
- with fused prepared chunk cache: `102.91 ms`
- speedup: about `1.39x`

The latest local pass also makes `3b` eligible for the direct torch prefill path instead of forcing it back through host encode+prepare. On a synthetic aligned MPS prefill prepare benchmark (`64` pages, `tokens_per_page=16`, `head_dim=64`):

- direct tensor-side `3b` prepare: `66.93 ms`
- encode on host plus prepare on MPS: `175.15 ms`
- speedup: about `2.62x`
- direct path host-to-device bytes: `0`
- encode-plus-prepare host-to-device bytes: `32,768`

That does not make `M0 3b` a real end-to-end model-path win yet, but it is a much stronger directional result for both MPS and CUDA: once low-bit static pages are sealed, a fused pre-scaled chunk representation is a better hot decode shape, and direct device-side spill packing is a much better prefill shape than bouncing `3b` pages back through host encode.

The next local follow-up tightened the grouped cached-decode shape too. On a synthetic grouped multi-query MPS microbench (`2` KV groups, `2` queries/group, `4` pages, `16` tokens/page, `head_dim=64`):

- `M0 3b`
  - grouped decode `10.99 ms`
  - prepared chunk cache resident bytes `135,168`
- `M0 4b`
  - grouped decode `9.91 ms`
  - prepared chunk cache resident bytes `135,168`

So the current local read is nuanced:

- the earlier “`3b` is faster than `4b`” read turned out to be mostly cache-policy shaped
- once grouped `4b` is allowed to use the same fused-only grouped cache on MPS, it recovers most of the gap and slightly edges out `3b` on this rerun
- both now pay the same grouped chunk-cache memory in this synthetic shape

That is a useful CUDA hint rather than a direct product claim: low-bit grouped fused caches can be worth it for decode, but the cache policy itself matters almost as much as the codec.

The new grouped low-bit microbench can now emit a small shape ladder instead of one hand-picked point. A compact MPS sweep over `bits={3,4}`, `page_count={2,4,8}`, and `query_count={1,2}` at the same grouped `64`-dim shape showed:

- once `3b` and `4b` use the same grouped fused-cache policy, there is no clean universal winner on these tiny Apple workloads
- both bits now scale chunk-cache residency identically in this shape:
  - `67,584` bytes at `2` pages
  - `135,168` bytes at `4` pages
  - `270,336` bytes at `8` pages
- decode timings bounce around enough that the safe conclusion is structural, not absolute:
  - cache policy was a large part of the earlier `3b` vs `4b` difference
  - the remaining codec-only gap on MPS is small and noisy at this scale

So the best local read is:

- grouped fused caching is the important idea
- `3b` is still valuable because it opens a real extra planning tier
- but for grouped decode throughput, we should not overfit to one small MPS shape and declare `3b` or `4b` the universal winner

## Local M3 int8 Probe

`M3` now supports an `int8` escape path with per-row scales on the local MPS runtime. This keeps the `M3` live-tail semantics the same while reducing resident bytes for recent pages.

TinyLlama exact `10` prompt tokens is a clean local isolation point because the whole KV lives in the tail:

- `M3 float16`
  - dense decode `691.90 ms/step`
  - DotCache decode `3292.46 ms/step`
  - tail resident bytes `5.77 MB`
  - greedy agreement `1.0`
- `M3 int8`
  - dense decode `733.08 ms/step`
  - DotCache decode `3743.42 ms/step`
  - tail resident bytes `2.97 MB`
  - greedy agreement `1.0`

So the first-order trade is straightforward:

- resident tail memory dropped by about `48%`
- runtime got a bit worse on this MPS implementation
- short-prompt greedy behavior stayed unchanged

The latest local cleanup trims a bit of avoidable MPS overhead without changing `M3 int8` form:

- prepared `int8` escape scales now stay in `fp32` on MPS, so decode no longer widens them every step
- single-page `M3` decode skips an unnecessary stack/concat path

On a small synthetic single-page MPS decode microbench (`16` tokens, `head_dim=64`), the current shape is:

- `M3 float16`: `208.97 ms`
- `M3 int8`: `220.09 ms`

So `M3 int8` is still slower than `float16` on this Apple path, but the remaining gap now looks more like core dequant cost than obvious metadata churn.

A short TinyLlama teacher-forced check (`40 / 32 / 8`) with `M3 int8` also stayed clean enough to keep exploring:

- loss delta `-0.01082`
- token agreement `1.0`
- target match `1.0`

That makes `M3 int8` a plausible memory-first live-tail option. It is not yet a speed win on this Mac, but it is the first quantized `M3` path worth carrying forward into later planner and CUDA work.

## CUDA Supported Matrix Baseline

Current CUDA-supported baseline on this pod was recorded from:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_model_matrix.py \
  --run-supported \
  --backend torch_cuda \
  --device cuda \
  --max-new-tokens 2 \
  --output-format jsonl
```

Raw artifact:

- `benchmarks/results/cuda_supported_baseline_20260328_svd_shared.jsonl`

All successful exact-length HF runs kept greedy token agreement at `1.0`.

This refreshed checkpoint includes the later `Qwen2.5 7B` low-memory lane change too: the matrix now exercises planner-aggressive keys with `M4/project` on the `svd_shared` basis for that model, instead of the older `M2` planner lane.

| Model | Exact prompt | Decode ms/step | KV resident bytes | Total resident bytes | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `TinyLlama 1.1B Chat` | `289` | `91.56` | `7,557,120` | `11,206,656` | pass |
| `TinyLlama 1.1B Chat` | `577` | `100.35` | `9,340,032` | `13,935,744` | pass |
| `TinyLlama 1.1B Chat` | `1536` | `122.55` | `16,479,872` | `24,589,952` | pass |
| `SmolLM2 360M Instruct` | `1024` | `289.11` | `24,231,552` | `35,044,992` | pass |
| `SmolLM2 360M Instruct` | `2048` | `355.57` | `37,768,960` | `55,915,264` | pass |
| `SmolLM2 1.7B Instruct` | `1024` | `766.13` | `125,829,120` | `142,860,288` | pass |
| `SmolLM2 1.7B Instruct` | `2048` | `933.75` | `201,326,592` | `235,388,928` | pass |
| `Llama 3.2 3B Instruct` | `1024` | `216.89` | `73,400,320` | `109,576,192` | pass |
| `Llama 3.2 3B Instruct` | `2048` | `329.72` | `117,440,512` | `175,636,480` | pass |
| `Llama 3.2 3B Instruct` | `4096` | `452.94` | `205,520,896` | `271,581,184` | pass |
| `Qwen2.5 1.5B Instruct` | `1024` | `96.61` | `18,677,760` | `27,918,336` | pass |
| `Qwen2.5 1.5B Instruct` | `2048` | `110.78` | `30,015,488` | `44,957,696` | pass |
| `Qwen2.5 3B Instruct` | `1024` | `133.91` | `24,084,480` | `35,979,264` | pass |
| `Qwen2.5 3B Instruct` | `2048` | `146.61` | `38,731,776` | `57,802,752` | pass |
| `Qwen2.5 3B Instruct` | `4096` | `194.09` | `68,026,368` | `101,449,728` | pass |
| `Qwen2.5 7B Instruct` | `1024` | `164.48` | `33,154,816` | `48,113,408` | pass |
| `Qwen2.5 7B Instruct` | `2048` | `197.90` | `51,612,672` | `77,298,688` | pass |
| `Qwen2.5 7B Instruct` | `4096` | `267.79` | `88,529,920` | `132,117,504` | pass |

Current limits from the same matrix:

- GGUF external lanes were not runnable here because the required executable was missing

The practical read is:

- the full current HF CUDA matrix now clears through `Qwen2.5 7B @ 4096`
- the recommended low-memory `Qwen2.5 7B` CUDA lane is now the planner-aggressive `svd_shared` `M4/V4` path
- that 7B lane holds agreement `1.0` while cutting the `4096` KV footprint to `88,529,920` bytes
- `TinyLlama` still works as the smallest exact regression lane, but it is no longer the most representative performance target
- the next frontier issue is no longer basic CUDA viability; it is performance work on the heaviest exact lanes, especially `SmolLM2 1.7B` and the larger-context `Llama 3.2 3B` / `Qwen2.5 7B` paths

## 2026-03-28 - Qwen2.5 7B per-page planner policy on CUDA

I checked whether the current `Qwen2.5 7B` CUDA lane was already using the newer per-page selectable policy machinery. It was not. The existing selective wrapper still uses fixed layer and KV-group overrides:

- `layer:0=M3`
- `layer:27:kv:1=M3`

The real planner-driven benchmark path is the same `bench_qwen2_compare.py` harness with `--key-policy-tier ...` and no hardcoded `--key-mode-override` flags.

Exact `4096` comparison on CUDA with `--max-new-tokens 2`:

| Policy | Agreement | Decode ms/step | KV resident bytes | Total resident bytes | Key page mix |
| --- | ---: | ---: | ---: | ---: | --- |
| exact K (`K=M3 / V=M0`) | `1.00` | `150.22` | `176,160,768` | `220,200,960` | `1792` `K:M3` |
| fixed selective overrides | `1.00` | `266.84` | `106,037,248` | `158,072,832` | `80` `K:M3`, `1712` `K:M0` |
| planner `strict` | `0.50` | `324.38` | `102,760,448` | `153,092,096` | `1792` `K:M0` |
| planner `balanced` | `1.00` | `499.95` | `95,701,504` | `143,174,144` | `1178` `K:M2`, `599` `K:M0:4`, `15` `K:M0:2` |
| planner `aggressive` | `1.00` | `307.50` | `91,995,136` | `137,968,640` | `1692` `K:M2`, `98` `K:M0:2`, `2` `K:M0:4` |

The useful result is:

- plain planner `strict` is not enough for `Qwen2.5 7B`; it effectively collapses to all-`M0` and loses agreement
- planner `balanced` and `aggressive` both keep `1.0` agreement with true per-page adaptive key selection
- planner `aggressive` is the best current low-memory planner tier on this pod: it beats the fixed selective wrapper on KV bytes while staying materially faster than planner `balanced`
- the recommended default runtime lane does not change; exact-K remains much faster than any current adaptive planner policy at `4096`

So the repo now has two distinct `Qwen2.5 7B` CUDA stories:

- best runtime lane: `K=M3 / V=M0`
- best low-memory adaptive lane: planner `aggressive` per-page keys on top of `K=M0 / V=M0`

## 2026-03-28 - Cached grouped M2 tensors on CUDA

The next CUDA optimization target after landing the planner-aggressive `Qwen2.5 7B` lane was the `M2` key score path itself. The exact `4096` planner-aggressive profile made the bottleneck clear:

- `decode_score_ms_per_step = 130.19`
- `decode_mix_ms_per_step = 112.51`
- `decode_unpack_ms_per_step = 4.66`

So this was not an unpack problem. The hot path was repeatedly rebuilding grouped `M2` score tensors (`m2_sketch`, `m2_basis`, `m2_mean`) from individual prepared pages inside every decode step.

I fixed that by extending the prepared chunk caches in [torch_mps.py](/workspace/DotCache/dotcache/backends/torch_mps.py) to support `M2` pages on CUDA:

- single prepared chunks can now cache per-group stacked `M2` sketch, basis, and mean tensors
- grouped prepared chunks can now cache the batched grouped `M2` view used by grouped decode
- the CUDA `M2` score path now reuses those cached tensors instead of restacking them every layer

Validation:

- [test_torch_cuda_backend.py](/workspace/DotCache/tests/test_torch_cuda_backend.py): grouped CUDA `M2` decode slices still pass
- [test_vllm_adapter.py](/workspace/DotCache/tests/test_vllm_adapter.py): grouped prepared chunk cache now has explicit `M2` CUDA coverage

Measured on `Qwen/Qwen2.5-7B-Instruct`, exact `4096`, planner-aggressive, CUDA:

| Run | Decode ms/step | Score ms/step | Mix ms/step | Unpack ms/step | KV resident bytes | Total resident bytes | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before cached grouped `M2` | `307.50` | `130.19` | `112.51` | `4.66` | `91,995,136` | `137,968,640` | `1.00` |
| after cached grouped `M2` | `304.79` | `127.64` | `110.55` | `4.27` | `91,995,136` | `136,784,896` | `1.00` |

This is a real but modest win. The grouped `M2` path is still fundamentally score-heavy, but the chunk-cache extension did trim some decode-time orchestration overhead without changing the model-facing behavior.

I then pushed one step further on the same path: when the grouped prepared chunk cache is available, it now also materializes batched all-group `M2` tensors, and the grouped CUDA scorer uses those directly instead of looping over the `4` groups in Python.

Measured again on `Qwen/Qwen2.5-7B-Instruct`, exact `4096`, planner-aggressive, CUDA:

| Run | Decode ms/step | Score ms/step | Mix ms/step | Unpack ms/step | KV resident bytes | Total resident bytes | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after cached grouped `M2` | `304.79` | `127.64` | `110.55` | `4.27` | `91,995,136` | `136,784,896` | `1.00` |
| after batched grouped `M2` score | `297.56` | `116.63` | `109.76` | `4.78` | `91,995,136` | `136,072,192` | `1.00` |

This second step is worth keeping. It trims another `7.23 ms/step` off the real exact-length decode path and takes a meaningful bite out of the `M2` score band without increasing KV residency.

I also tried replacing the single-segment grouped `M2` score branch with a flattened batched-`matmul` formulation instead of the cached-tensor `einsum` path. That was a clear regression on the same workload, so I reverted it:

| Run | Decode ms/step | Score ms/step | Mix ms/step | Unpack ms/step | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept batched grouped `M2` score | `297.56` | `116.63` | `109.76` | `4.78` | `1.00` |
| reverted flattened `matmul` rewrite | `354.51` | `125.67` | `117.86` | `6.05` | `1.00` |

So the next useful CUDA step is not another PyTorch-level reshaping of the single-segment `M2` score math. The remaining win, if there is one, likely needs a dedicated lower-level kernel or a different page-native formulation.

I also tried that next lower-level step directly: a guarded CUDA-only Triton scorer for the exact single-segment grouped `M2` lane used by `Qwen2.5 7B` planner-aggressive (`group_size=32`, `num_groups=4`, `rank=8`). The result was mixed and not good enough to keep, so I reverted it.

| Run | Decode ms/step | Score ms/step | Mix ms/step | Unpack ms/step | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept cached grouped `M2` path | `297.56` | `116.63` | `109.76` | `4.78` | `1.00` |
| reverted Triton `M2` scorer, profiled | `302.55` | `114.48` | `109.30` | `4.75` | `1.00` |
| reverted Triton `M2` scorer, plain rerun A | `306.87` | n/a | n/a | n/a | `1.00` |
| reverted Triton `M2` scorer, plain rerun B | `317.92` | n/a | n/a | n/a | `1.00` |

So even though the profiled score band improved slightly, the real exact-length end-to-end decode regressed. That rules out this first Triton score-only shape as the next keepable CUDA path.

I also tried a decode-structure change instead of a new kernel: routing `decode_layer_torch` through the grouped output-only path only for the CUDA adaptive-key case (`M2` keys with `M0` values), since that path already does online softmax and only needs the final activations. That also turned out to be a dead end, so I reverted it.

| Run | Decode ms/step | Score ms/step | Mix ms/step | Unpack ms/step | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| kept cached grouped `M2` path | `297.56` | `116.63` | `109.76` | `4.78` | `1.00` |
| reverted selective output-only route, profiled | `308.45` | `117.41` | `110.88` | `4.72` | `1.00` |
| reverted selective output-only route, plain | `314.63` | n/a | n/a | n/a | `1.00` |

So the next useful CUDA step is narrower again: not a score-only kernel, and not switching the adaptive lane over to the existing output-only decode structure. The remaining win likely needs a more integrated fused design or a different adaptive page representation.

## 2026-03-28 - Experimental fixed-project key pages (`M4`) for CUDA adaptive Qwen7B

The next experiment after exhausting the obvious `M2/sketch` kernel and decode-structure ideas was a different adaptive key representation altogether. I added an experimental key-only mode, `M4/project`, behind an explicit config flag:

- it keeps the planner shape the same as the current aggressive adaptive lane
- it swaps the `M2/sketch` candidate for a fixed-basis projected key page
- each key page stores per-token coefficients against a shared Hadamard-derived basis plus one page mean per group
- score no longer needs page-specific basis application; it becomes a projected-query dot product against those stored coefficients

This mode is intentionally guarded by `prefer_m4_project_k=True`; it does not affect the current default planner behavior.

Validation:

- [test_m4_key_project.py](/workspace/DotCache/tests/test_m4_key_project.py): CPU/reference coverage for `M4/project`
- [test_torch_cuda_backend.py](/workspace/DotCache/tests/test_torch_cuda_backend.py): CUDA decode parity for `M4` key pages
- [test_vllm_adapter.py](/workspace/DotCache/tests/test_vllm_adapter.py): grouped prepared-chunk cache coverage for CUDA `M4` pages

Measured on `Qwen/Qwen2.5-7B-Instruct`, exact `4096`, planner-aggressive, CUDA, same command shape:

| Run | Decode ms/step | Score ms/step | Mix ms/step | Unpack ms/step | KV resident bytes | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| planner aggressive `M2/sketch`, plain | `334.83` | n/a | n/a | n/a | `89,400,576` | `1.00` |
| planner aggressive `M4/project`, plain | `311.43` | n/a | n/a | n/a | `88,227,072` | `1.00` |
| planner aggressive `M2/sketch`, profiled | `330.62` | `138.49` | `121.63` | `1.97` | `89,400,576` | `1.00` |
| planner aggressive `M4/project`, profiled | `337.30` | `133.62` | `127.02` | `2.22` | `88,227,072` | `1.00` |

So the current read is mixed but useful:

- the plain exact-length run improved materially with `M4/project`
- KV residency also improved slightly
- the profiled run stayed in roughly the same overall band
- score improved a bit, but that win was partly offset elsewhere, so this is not yet a clear decode-kernel breakthrough

That makes `M4/project` worth keeping as an experimental lane, but not worth promoting over `M2/sketch` yet. It gives the repo a reproducible alternative adaptive key representation to iterate on, and it is simple enough that future work can target either:

- better fixed-basis choices / projection rank, or
- a dedicated fast path for `M4/project` if the plain-path improvement proves stable

I then reran the `M4/project` lane after merging the newer `main` branch, which now includes the recent MPS low-bit follow-ups and planner updates, and did a small exact `4096` stability sweep on the CUDA `Qwen2.5 7B` planner-aggressive lane:

| Run | Decode ms/step | KV resident bytes | Agreement |
| --- | ---: | ---: | ---: |
| `M2/sketch` baseline (`rank=8`) | `347.49` | `89,400,576` | `1.00` |
| `M4/project` `rank=4` | `340.82` | `83,533,056` | `1.00` |
| `M4/project` `rank=8` run A | `326.80` | `88,227,072` | `1.00` |
| `M4/project` `rank=8` run B | `335.98` | `88,227,072` | `1.00` |
| `M4/project` `rank=16` | `336.71` | `97,615,104` | `1.00` |

That sweep tightened the conclusion:

- `M4/project` is still a real keepable experimental lane
- `rank=8` remains the best current `M4` setting on this pod
- `rank=4` saves more memory but gives up some decode time
- `rank=16` increases memory without paying it back in speed
- the two `rank=8` runs show enough variance that `M4` should stay experimental for now rather than replacing `M2` in the planner defaults

The next question was whether the new `main` branch `M0` 3-bit work helps if we pair it with the experimental `M4/project` key lane on CUDA. I ran a three-way exact-length sweep on `Qwen/Qwen2.5-7B-Instruct`, planner-aggressive keys, exact values, CUDA:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen2_compare.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --default-mode-k M0 \
  --default-mode-v M0 \
  --key-policy-tier aggressive \
  --value-policy-tier exact \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 4096 \
  --max-new-tokens 2
```

The three compared lanes were:

- baseline adaptive lane: planner-aggressive `K=M2/sketch`, `V=M0` 4-bit
- experimental fast adaptive lane: planner-aggressive `K=M4/project rank=8`, `V=M0` 4-bit
- experimental low-memory lane: planner-aggressive `K=M4/project rank=8`, `V=M0` 3-bit

| Prompt | Lane | Decode ms/step | KV resident bytes | Agreement |
| --- | --- | ---: | ---: | ---: |
| `1024` | `M2/V4` | `215.45` | `33,361,920` | `1.00` |
| `1024` | `M4/V4` | `228.96` | `33,067,008` | `1.00` |
| `1024` | `M4/V3` | `269.84` | `31,232,000` | `1.00` |
| `2048` | `M2/V4` | `242.44` | `52,043,776` | `1.00` |
| `2048` | `M4/V4` | `226.51` | `51,453,952` | `1.00` |
| `2048` | `M4/V3` | `273.78` | `47,783,936` | `1.00` |
| `4096` | `M2/V4` | `355.10` | `89,400,576` | `1.00` |
| `4096` | `M4/V4` | `302.23` | `88,227,072` | `1.00` |
| `4096` | `M4/V3` | `345.53` | `80,887,040` | `1.00` |

The page mix stayed structurally the same across the three lanes; the planner was deciding the key pages and values remained all `M0` pages. At exact `4096`:

- `M2/V4`: `1219` `K:M0:2`, `573` `K:M2:4`, `1792` `V:M0`
- `M4/V4`: `1219` `K:M0:2`, `573` `K:M4:4`, `1792` `V:M0`
- `M4/V3`: `1219` `K:M0:2`, `573` `K:M4:4`, `1792` `V:M0`

That sweep gives the next working split:

- best adaptive-performance lane: `M4/project rank=8` with `V=M0` 4-bit
- best adaptive-min-memory lane: `M4/project rank=8` with `V=M0` 3-bit
- current baseline to beat: `M2/sketch` with `V=M0` 4-bit

The practical conclusion is that the new 3-bit value support is useful here as a memory tier, but not as the default adaptive CUDA lane. It trims about `7.34 MiB` of KV memory at exact `4096`, while giving back most of the decode-time win from `M4/project`.

I then tested whether that `M4/project` win was strong enough to replace the existing shared `Qwen2.5 7B` adaptive CUDA surfaces, using the same natural-text prompt unit the shared model matrix uses:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen2_compare.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --default-mode-k M0 \
  --default-mode-v M0 \
  --key-policy-tier aggressive \
  --value-policy-tier exact \
  --target-prompt-lengths 2048 4096 \
  --max-new-tokens 2 \
  --prompt-unit 'Cache locality matters for fast decoding.'
```

That surfaced a real blocker:

| Prompt | Lane | Decode ms/step | KV resident bytes | Agreement |
| --- | --- | ---: | ---: | ---: |
| `2048` | planner-aggressive `M2/V4` | `211.69` | `53,341,184` | `1.00` |
| `2048` | planner-aggressive `M4/V4` | `189.69` | `51,612,672` | `0.50` |
| `4096` | planner-aggressive `M2/V4` | `276.18` | `91,995,136` | `1.00` |
| `4096` | planner-aggressive `M4/V4` | `252.82` | `88,529,920` | `0.50` |

So `M4/project` is faster on this prompt shape too, but it is not robust enough to replace the shared adaptive lane. The current repo-wide recommendation stays:

- shared adaptive `Qwen2.5 7B` CUDA lane: planner-aggressive `M2/sketch` with `V=M0` 4-bit
- experimental faster adaptive lane: planner-aggressive `M4/project` with `V=M0` 4-bit
- experimental minimum-memory lane: planner-aggressive `M4/project` with `V=M0` 3-bit

That means `M4` stays an explicit research wrapper for now rather than becoming the default matrix or wrapper path.

I then localized that prompt-shape failure one step further on the same natural-text `2048` prompt. The high-level compare harness already showed that the generated text diverges immediately after the first shared continuation token:

- passing shared adaptive lane (`M2/V4`): `" fast decoding"`
- failing experimental lane (`M4/V4`): `" fast,"`

To separate “first generated token” from “first replay decode step”, I ran a focused replay-style probe on the exact same `2048` prompt. In this probe, `step 0` corresponds to the first decode step *after* the first generated token has already been appended to the cache. That makes the result line up cleanly with the harness output above:

- the first generated token still matches
- the divergence begins on the next decode step
- the largest replay-context drift is concentrated in the final decoder layers rather than being spread evenly across the stack

At `2048`, the passing `M2/V4` lane still has the same top-error layers, but the errors stay bounded enough to keep token agreement:

| Lane | Replay step | Top layers by max abs context error |
| --- | ---: | --- |
| `M2/V4` | `0` | `27: 9.13`, `21: 6.58`, `26: 5.56`, `11: 5.43`, `24: 4.54` |
| `M2/V4` | `1` | `26: 8.90`, `24: 5.48`, `21: 5.48`, `27: 5.33`, `11: 4.66` |

On the failing `M4/V4` lane, those same late layers blow up more sharply, especially `26` and `27`:

| Lane | Replay step | Top layers by max abs context error |
| --- | ---: | --- |
| `M4/V4` | `0` | `26: 17.73`, `27: 13.35`, `23: 8.90`, `25: 7.26`, `24: 6.57` |
| `M4/V4` | `1` | `26: 16.78`, `27: 11.41`, `25: 7.92`, `23: 7.32`, `24: 5.85` |

That gives the next debugging target a much tighter shape:

- the problem is not “early-layer global drift”
- it is not a value-side issue; values were unchanged
- the failure is most visible in the late attention stack, especially layers `26` and `27`
- the first bad token on the natural-text prompt appears immediately after the first shared generated token

So the next real `M4` task is to understand why the fixed-project key representation is underestimating or mis-ranking late-layer attention on the continuation step, not to keep changing repo-wide defaults.

I then checked that directly at the page-score level on the same failing `2048` natural-text prompt, focusing on the late layers that dominated the replay drift (`23`-`27`) and keeping values exact so this stayed a key-side comparison. The result is clear: `M4` is not merely adding larger-but-rank-stable score noise. It is actually reordering attention across pages in the late layers.

For the passing `M2` lane, late-layer page-score quality is imperfect but still mostly aligned:

| Layer | Step | `M2` top-1 match | `M2` top-k overlap | `M2` KL |
| --- | ---: | ---: | ---: | ---: |
| `23` | `0` | `0.86` | `0.72` | `0.09` |
| `24` | `0` | `1.00` | `0.69` | `0.13` |
| `25` | `0` | `0.75` | `0.75` | `0.29` |
| `26` | `0` | `0.89` | `0.79` | `0.10` |
| `27` | `0` | `0.86` | `0.74` | `0.11` |

On the failing `M4` lane, those same layers lose page-order fidelity sharply:

| Layer | Step | `M4` top-1 match | `M4` top-k overlap | `M4` KL |
| --- | ---: | ---: | ---: | ---: |
| `23` | `0` | `0.29` | `0.41` | `3.29` |
| `24` | `0` | `0.36` | `0.30` | `3.17` |
| `25` | `0` | `0.36` | `0.54` | `2.87` |
| `26` | `0` | `0.29` | `0.33` | `3.61` |
| `27` | `0` | `0.64` | `0.38` | `2.12` |

That confirms the failure mode:

- `M4` is mis-ranking key-page attention in the late layers
- the worst layers are still `26` and `27`, but the degradation starts earlier in the same late block
- narrow layer rescues (`26`-`27`, then `23`-`27`) are not enough to restore token agreement

So the next `M4` debugging step should target the representation itself, especially how the fixed-project key pages preserve score ordering in late Qwen2.5 7B layers, not just another wrapper policy tweak.

I then added explicit `M4` basis-family support so the encoded page carries its own project basis (`hadamard` or `dct`), with optional per-layer overrides, and reran the same failing natural-text `Qwen2.5 7B` CUDA prompt to test whether the instability was specific to the current Hadamard-style basis.

Validation for the basis-family plumbing passed:

- `python -m py_compile dotcache/types.py dotcache/modes/m4_key_project.py dotcache/config.py dotcache/encode.py dotcache/decode_reference.py dotcache/session_runtime.py dotcache/attention_reference.py dotcache/backends/torch_mps.py benchmarks/bench_qwen2_compare.py tests/test_m4_key_project.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_m4_key_project.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_torch_cuda_backend.py -k 'project_m4_keys_decode_on_cuda or segmented_m2_keys_decode_on_cuda'`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_vllm_adapter.py -k 'grouped_prepared_chunk_cache_builds_m4_cuda_view or grouped_prepared_chunk_cache_builds_m2_cuda_view'`

The real benchmark result is negative. Swapping late layers `23`-`27` to a DCT basis did not recover agreement, and a global DCT basis was slightly worse:

| Prompt | Lane | Decode ms/step | KV resident bytes | Agreement |
| --- | --- | ---: | ---: | ---: |
| `2048` | `M4/V4` Hadamard baseline | `189.69` | `51,612,672` | `0.50` |
| `2048` | `M4/V4` DCT on layers `23`-`27` | `194.03` | `51,612,672` | `0.50` |
| `2048` | `M4/V4` DCT global | `208.84` | `51,612,672` | `0.50` |
| `4096` | `M4/V4` Hadamard baseline | `252.82` | `88,529,920` | `0.50` |
| `4096` | `M4/V4` DCT on layers `23`-`27` | `256.55` | `88,529,920` | `0.50` |
| `4096` | `M4/V4` DCT global | `272.42` | `88,529,920` | `0.50` |

That narrows the diagnosis again:

- the `M4` failure is not specific to the Hadamard basis family
- simply swapping to another fixed orthogonal basis does not restore page-score fidelity
- the problem is more structural than "pick a better fixed basis"

So the next plausible `M4` move is no longer "try another fixed basis." It should be a richer representation, such as a learned/data-driven basis per layer or a representation that stores page-local basis information instead of assuming one global fixed family.

I then implemented the smallest richer variant directly in `M4`: a page-local learned basis (`project_basis=svd`) stored alongside the coefficients, keeping the same `M4` mode surface but letting each page carry its own low-rank basis instead of assuming a fixed global one.

The focused implementation validation passed:

- `python -m py_compile dotcache/modes/m4_key_project.py dotcache/encode.py dotcache/attention_reference.py dotcache/decode_reference.py dotcache/session_runtime.py dotcache/backends/torch_mps.py benchmarks/bench_qwen2_compare.py tests/test_m4_key_project.py tests/test_torch_cuda_backend.py tests/test_vllm_adapter.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_m4_key_project.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_torch_cuda_backend.py -k 'project_m4_keys_decode_on_cuda or segmented_m2_keys_decode_on_cuda'`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_vllm_adapter.py -k 'grouped_prepared_chunk_cache_builds_m4_cuda_view or grouped_prepared_chunk_cache_builds_m2_cuda_view or grouped_prepared_chunk_cache_builds_m4_svd_cuda_view'`

On the same natural-text `Qwen/Qwen2.5-7B-Instruct` CUDA prompt that fixed-basis `M4` kept failing on, the learned-basis `M4` path restores agreement:

| Prompt | Lane | Decode ms/step | KV resident bytes | Agreement |
| --- | --- | ---: | ---: | ---: |
| `2048` | `M2/V4` | `211.69` | `53,341,184` | `1.00` |
| `2048` | fixed-basis `M4/V4` | `189.69` | `51,612,672` | `0.50` |
| `2048` | learned-basis `M4/V4` (`svd`) | `202.16` | `53,341,184` | `1.00` |
| `4096` | `M2/V4` | `276.18` | `91,995,136` | `1.00` |
| `4096` | fixed-basis `M4/V4` | `252.82` | `88,529,920` | `0.50` |
| `4096` | learned-basis `M4/V4` (`svd`) | `272.47` | `91,995,136` | `1.00` |

That changes the diagnosis:

- the failure is specific to fixed-basis `M4`, not to the broader "projected coefficients + mean" structure
- a page-local learned basis is enough to recover the late-layer score ordering on the failing natural-text prompt
- but the learned-basis `M4` lane gives back essentially all of the fixed-basis memory win, landing on the same KV footprint as `M2`

So the next meaningful representation goal is now clearer: not "another fixed basis" and not "plain learned basis." To beat `M2`, the likely target is a compressed learned basis, a layer-shared learned basis, or another low-overhead representation that preserves page-score ordering without storing a full page-local basis tensor.

I then implemented that next step as `M4` with a learned shared basis (`project_basis=svd_shared`): one learned basis per layer and KV head, shared across pages, while keeping page-local coefficients and means. The first pass exposed two CUDA-side issues rather than a representation failure:

- grouped CUDA decode incorrectly collapsed the per-head shared-basis axis inside the grouped prepared-chunk cache
- the single-head CUDA fallback path dropped from the learned shared basis back to the fixed-basis `M4` path whenever `PreparedPageTorch.m2_basis` was intentionally left `None`

After fixing both of those, the focused CUDA validation passed:

- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_torch_cuda_backend.py -k 'project_m4_svd_shared_keys_decode_on_cuda or project_m4_keys_decode_on_cuda or segmented_m2_keys_decode_on_cuda'`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_vllm_adapter.py -k 'grouped_prepared_chunk_cache_builds_m4_svd_shared_cuda_view or grouped_prepared_chunk_cache_builds_m4_svd_cuda_view or grouped_prepared_chunk_cache_builds_m4_cuda_view'`

On the same natural-text `Qwen/Qwen2.5-7B-Instruct` CUDA prompt shape that fixed-basis `M4` failed on, `svd_shared` now recovers agreement while also keeping the low-memory `M4` footprint:

| Prompt | Lane | Decode ms/step | KV resident bytes | Agreement |
| --- | --- | ---: | ---: | ---: |
| `2048` | `M2/V4` | `211.69` | `53,341,184` | `1.00` |
| `2048` | fixed-basis `M4/V4` | `189.69` | `51,612,672` | `0.50` |
| `2048` | learned page-local `M4/V4` (`svd`) | `202.16` | `53,341,184` | `1.00` |
| `2048` | learned shared-basis `M4/V4` (`svd_shared`) | `198.27` | `51,612,672` | `1.00` |
| `4096` | `M2/V4` | `276.18` | `91,995,136` | `1.00` |
| `4096` | fixed-basis `M4/V4` | `252.82` | `88,529,920` | `0.50` |
| `4096` | learned page-local `M4/V4` (`svd`) | `272.47` | `91,995,136` | `1.00` |
| `4096` | learned shared-basis `M4/V4` (`svd_shared`) | `267.79` | `88,529,920` | `1.00` |

That is the first `M4` variant that actually clears the target tradeoff:

- it keeps the fixed-basis `M4` memory advantage
- it restores the natural-text agreement failure that fixed-basis `M4` had at `2048` and `4096`
- it stays faster than `M2/V4`, though not as fast as the unstable fixed-basis `M4`

So the next repo-level move is no longer representation discovery. It is productization: promote `svd_shared` as the experimental low-memory adaptive `Qwen2.5 7B` CUDA lane, and then rerun the broader matrix or wrapper surface on top of that corrected basis-sharing path.

## 2026-03-28 13:29 UTC - GGUF / llama.cpp CUDA reference baseline

I finished hardening the GGUF reference path on the pod:

- `llama-cli` is installed from `/workspace/llama.cpp`
- the runner prefers persistent local files under `/workspace/models/gguf`
- the dead `ggml-org/Llama-3.2-3B-Instruct-GGUF` repo ID is replaced with `bartowski/Llama-3.2-3B-Instruct-GGUF`
- the Qwen GGUF lanes now pin explicit `Q4_K_M` files too

I attempted the existing exact-prompt `bench_gguf_external.py` matrix first, but it is a poor fit for full-pod baseline collection here because `llama-cli` startup and prompt-eval cost dominate and make the long-form exact harness too expensive to complete across all three models in one pass. For the actual GGUF CUDA checkpoint, I switched to `llama-bench`, which is the right reference tool for raw `llama.cpp` throughput.

Command:

- `source scripts/env_cuda.sh && {`
- `llama-bench -m /workspace/models/gguf/llama32_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 1024,2048,4096 -n 2 -r 1 -o jsonl;`
- `llama-bench -m /workspace/models/gguf/qwen25_3b/qwen2.5-3b-instruct-q4_k_m.gguf -p 1024,2048,4096 -n 2 -r 1 -o jsonl;`
- `llama-bench -m /workspace/models/gguf/qwen25_7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -p 1024,2048,4096 -n 2 -r 1 -o jsonl;`
- `} > benchmarks/results/gguf_reference_bench_20260328.jsonl`

Raw artifact:

- [benchmarks/results/gguf_reference_bench_20260328.jsonl](/workspace/DotCache/benchmarks/results/gguf_reference_bench_20260328.jsonl)

All rows are `CUDA` backend with `n_gpu_layers=99`, `type_k=f16`, and `type_v=f16`.

| Model | Prompt tok/s @1024 | Prompt tok/s @2048 | Prompt tok/s @4096 | Decode tok/s @n=2 |
| --- | ---: | ---: | ---: | ---: |
| `Llama 3.2 3B Instruct Q4_K_M` | `22014.87` | `20106.80` | `16518.82` | `225.75` |
| `Qwen2.5 3B Instruct Q4_K_M` | `21209.22` | `19723.76` | `16761.18` | `190.97` |
| `Qwen2.5 7B Instruct Q4_K_M` | `13734.35` | `12480.03` | `10766.71` | `169.80` |

The useful practical baseline is:

- prompt throughput stays high on the RTX 5090, even at `4096`, ranging from about `10.8k` to `22.0k tok/s`
- decode throughput is much lower and scales with model size, ranging from about `169.8` to `225.8 tok/s`
- the GGUF reference lane is now genuinely runnable and locally cached, but the right baseline tool for it is `llama-bench`, not the exact-text `llama-cli` harness

## 2026-03-28 16:20 UTC - Qwen3.5 0.8B CUDA fast path and attention-subset third pass

I installed the missing native Qwen3.5 fast-path dependencies into the repo `.venv`:

- `flash-linear-attention==0.4.2`
- `causal-conv1d==1.6.1` built with `--no-build-isolation` against local `torch 2.8.0+cu128`

That flipped the Transformers gate from fallback-Torch mode to the intended native path:

- `is_flash_linear_attention_available=True`
- `is_causal_conv1d_available=True`
- `transformers.models.qwen3_5.modeling_qwen3_5.is_fast_path_available=True`

The dense CUDA baseline improved materially once that path was live:

- exact `512` prompt:
  - before: prefill `393.57 ms`, decode `21.99 ms/step`
  - after: prefill `26.63 ms`, decode `16.25 ms/step`
- repeat `64` prompt (`448` tokens):
  - before: prefill `663.60 ms`, decode `93.60 ms/step`
  - after: prefill `26.93 ms`, decode `16.86 ms/step`

With the dense baseline fixed, I reran the attention-subset CUDA policy probes. The Mac second-pass profile was still useful directionally, but it was not the best CUDA starter once the native fast path was active.

Second pass on CUDA:

- exact `64` prompt:
  - replay context max abs error: `0.2979`
  - replay output max abs error: `0.0740`
  - teacher-forced logit max abs error: `1.2695`
- repeat `32` prompt:
  - replay context max abs error: `0.3164`
  - replay output max abs error: `0.1157`
  - teacher-forced logit max abs error: `1.4531`

The best narrower CUDA third pass was:

- key strict:
  - `layer:11`
  - `layer:19`
- value policy override:
  - `layer:15=M0/affine/4,M0/affine/3,M1/lut/4`
- exact values:
  - `layer:19=M3`
  - `layer:23=M3`

That profile is now captured in:

- [configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml](/workspace/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml)

Its measured CUDA results were:

- exact `64` prompt:
  - replay context max abs error: `0.2076`
  - replay output max abs error: `0.0314`
  - teacher-forced logit max abs error: `0.7148`
  - decode `51.76 ms/step`
- repeat `32` prompt:
  - replay context max abs error: `0.2773`
  - replay output max abs error: `0.0498`
  - teacher-forced logit max abs error: `0.7441`
  - decode `95.22 ms/step`

The important policy read is:

- `layer:7` key strictness was not the right CUDA lever once the native Qwen3.5 fast path was active
- values at `19` and `23` are the main fidelity anchors on CUDA
- the best next step is still attention-subset fidelity and hybrid-state integration, but it should now start from the CUDA third-pass profile rather than the older Mac second pass

## 2026-03-28 17:10 UTC - Qwen3.5 hybrid-state partition on CUDA

I turned the earlier byte accounting into an explicit adapter-level hybrid-state partition for Qwen3.5:

- fixed resident state:
  - `linear_attention` conv state
  - `linear_attention` recurrent state
- token-growing state:
  - `full_attention` key/value cache only

The implementation now exposes that split through the Qwen3.5 adapter and the hybrid inspect bench, rather than leaving it as an implicit interpretation of raw cache bytes.

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py benchmarks/bench_qwen35_hybrid_inspect.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py`
  - result: `17 passed`

Live CUDA check:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_hybrid_inspect.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --target-prompt-lengths 64 \
  --max-new-tokens 2
```

Useful result at exact `64`, after `2` decode steps:

- prefill fixed resident bytes: `19,759,104`
- prefill token-growing bytes: `86,016`
- final fixed resident bytes: `19,759,104`
- final token-growing bytes: `110,592`
- fixed resident growth: `0`
- token-growing growth: `24,576`

Layer split on the live model:

- fixed resident layers:
  - `[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]`
- token-growing layers:
  - `[3, 7, 11, 15, 19, 23]`

So the next Qwen3.5 abstraction step is now concrete:

- keep linear-attention state as a fixed resident object
- only let the six full-attention subset layers participate in token-growing DotCache KV

## 2026-03-28 17:35 UTC - Qwen3.5 attention-subset runtime now carries an explicit native hybrid split

I moved the fixed-resident vs token-growing split from the inspect-only path into the actual attention-subset DotCache runtime.

The Qwen3.5 adapter now captures the post-handoff native hybrid partition at prefill time:

- fixed resident:
  - all `linear_attention` conv/recurrent state
- token growing:
  - only the `full_attention` subset layers, after their native KV has been replaced by placeholders and handed off to DotCache

That means the runtime contract is now explicit:

- native hybrid state keeps the linear-attention resident state alive across decode
- DotCache owns the token-growing KV for layers `[3, 7, 11, 15, 19, 23]`

Validation:

- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'dotcache_harness or hybrid_state'`
  - result: `6 passed`

## 2026-03-28 19:35 UTC - Qwen3.5 4B StateCache needs an early-layer M3 escape policy, not a tail-only rescue

I added per-layer recurrent-state mode overrides to the Qwen3.5 DeltaNet StateCache harnesses so the 4B lane can mix `M0` and `M3` on resident recurrent state.

Files:

- [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py)
- [bench_qwen35_deltanet_statecache_readout.py](/workspace/DotCache/benchmarks/bench_qwen35_deltanet_statecache_readout.py)
- [bench_qwen35_deltanet_statecache_loss.py](/workspace/DotCache/benchmarks/bench_qwen35_deltanet_statecache_loss.py)
- [test_qwen35_integration.py](/workspace/DotCache/tests/test_qwen35_integration.py)

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py benchmarks/bench_qwen35_deltanet_statecache_readout.py benchmarks/bench_qwen35_deltanet_statecache_loss.py tests/test_qwen35_integration.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'statecache_readout or statecache_loss'`
  - result: `5 passed`

The first mixed-policy teacher-forced `1024` comparison on `Qwen/Qwen3.5-4B` showed that late-layer escapes alone are not enough:

- baseline `post_update_m0`, `8-bit`, no renorm:
  - `teacher_forced_token_agreement_rate = 0.90625`
  - `teacher_forced_loss_delta = 0.0132`
- layers `28-30 = M3`:
  - agreement stayed `0.90625`
  - `teacher_forced_loss_delta = 0.0123`
- layers `24-26, 28-30 = M3`:
  - agreement stayed `0.90625`
  - `teacher_forced_loss_delta = 0.0117`

The real-state sweep explains why. On captured recurrent samples at exact `64`, `M0 8-bit` is actually worst at the head:

- layer `0`: `best_update_error = 0.0129`
- layer `16`: `best_update_error = 0.0021`
- layer `30`: `best_update_error = 0.0037`
- `renorm_interval = 0` remained best for all three sampled layers

That shifted the policy search to the early recurrent layers, and that fixed the 4B lane:

- teacher-forced `1024`, `32` eval steps, `post_update_m0`, `8-bit`, no renorm, recurrent `M3` on layers `0,1,2`:
  - `teacher_forced_token_agreement_rate = 1.0`
  - `teacher_forced_target_match_rate = 1.0`
  - `teacher_forced_loss_delta = 0.0015`
  - `teacher_forced_perplexity_ratio = 1.0015`
  - `deltanet_statecache_recurrent_state_bytes = 20,054,016`
  - `effective_recurrent_compression_ratio = 2.51x`

Adding late-layer escapes on top of that did not materially improve quality:

- layers `0,1,2,28,29,30`:
  - agreement stayed `1.0`
  - `teacher_forced_loss_delta = 0.0009`
- layers `0,1,2,24,25,26,28,29,30`:
  - agreement stayed `1.0`
  - `teacher_forced_loss_delta = 0.0022`

Exact generation at `1024` also stayed clean with the minimal early escape policy:

- layers `0,1,2 = M3`:
  - dense ids `[12482, 364, 4778, 45543]`
  - StateCache ids `[12482, 364, 4778, 45543]`
  - `greedy_token_agreement_rate = 1.0`
  - `deltanet_statecache_decode_ms_per_step = 22.46`
  - `deltanet_statecache_recurrent_state_bytes = 20,054,016`

So the 4B read is now concrete:

- the safe rescue is an early recurrent escape policy, not a late one
- the smallest keepable policy is recurrent `M3` on layers `0,1,2`
- that policy is strong enough to promote into the CUDA matrix surface

## 2026-03-28 19:45 UTC - Qwen3.5 now has a runnable 8-bit readout-only StateCache prototype

I turned the earlier DeltaNet ablation result into a real harness mode instead of leaving it as a journal-only conclusion.

New surface:

- [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py)
  - `run_qwen35_deltanet_statecache_readout_harness(...)`
  - `Qwen35DeltaNetStateHarness.run_deltanet_statecache_readout(...)`
- [bench_qwen35_deltanet_statecache_readout.py](/workspace/DotCache/benchmarks/bench_qwen35_deltanet_statecache_readout.py)

This prototype intentionally does one narrow thing:

- keep the native DeltaNet update path dense
- quantize only the recurrent-state readout input with `M0`
- report compressed fixed-resident bytes against the real native resident-state footprint

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py benchmarks/bench_qwen35_deltanet_statecache_readout.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state or statecache_readout'`
  - result: `6 passed`

Live CUDA prototype on `Qwen/Qwen3.5-0.8B`, exact `64`, `4` decode steps:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 64 --max-new-tokens 4 --bits 8 --continue-on-error`
- `deltanet_dense_fixed_resident_bytes = 19,759,104`
- `deltanet_statecache_fixed_resident_bytes = 6,782,976`
- `deltanet_recurrent_state_bytes = 18,874,368`
- `deltanet_statecache_recurrent_state_bytes = 5,898,240`
- recurrent compression ratio: `3.2x`
- fixed-resident compression ratio: `2.91x`
- `deltanet_statecache_output_max_abs_error = 0.0060`
- `deltanet_statecache_max_abs_error = 0.0256`
- `error_grows_step_to_step = false`

So the first real StateCache result is good enough to keep:

- `8-bit readout-only M0` is now a real runnable Qwen3.5 prototype
- it materially reduces resident-state bytes
- it stays in the same low-error band the earlier DeltaNet ablation predicted

The next StateCache step is now narrower:

- compare readout-only `8-bit` against a true generated-token quality metric, not just layer/output drift
- only after that, decide whether to try `post_update` or a mixed `readout_only + M3 escape` policy

## 2026-03-28 20:05 UTC - Readout-only StateCache keeps token quality but is still a Python-level prototype

I extended the readout-only harness to run a true model-level decode by feeding quantized DeltaNet recurrent state back into native `past_key_values` before each decode step.

That makes the result materially stronger than the earlier ablation-only checkpoint: it now reports generated-token agreement, not just layer/output drift.

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py benchmarks/bench_qwen35_deltanet_statecache_readout.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state or statecache_readout'`
  - result: `6 passed`

Live CUDA run on `Qwen/Qwen3.5-0.8B`, exact `64`, `4` generated tokens:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 64 --max-new-tokens 4 --bits 8 --continue-on-error`
- dense generated ids:
  - `[65789, 12482, 364, 4778]`
- StateCache generated ids:
  - `[65789, 12482, 364, 4778]`
- `deltanet_statecache_greedy_token_agreement_rate = 1.0`
- `deltanet_statecache_output_max_abs_error = 0.0060`
- `deltanet_statecache_max_abs_error = 0.0256`
- `deltanet_statecache_effective_recurrent_compression_ratio = 3.2`
- `deltanet_statecache_effective_fixed_resident_compression_ratio = 2.91`

After replacing the CPU/Numpy codec loop with a torch-native grouped affine quant/dequant path, the prototype runtime moved substantially:

- earlier Python-loop checkpoint:
  - `deltanet_statecache_decode_ms_per_step = 1162.74`
- current torch-native checkpoint:
  - `dense_decode_ms_per_step = 134.31`
  - `deltanet_statecache_decode_ms_per_step = 37.97`

So the quality question is answered for this narrow case:

- `8-bit` readout-only DeltaNet StateCache can preserve generated-token output at exact `64`
- the implementation is no longer bottlenecked by Python-side quantization

That changes the next StateCache priority again:

- keep this torch-native path as the working baseline
- validate it on longer prompts and a teacher-forced or generation-quality surface beyond exact `64`
- only after that decide whether to add a resident packed representation or move to `post_update` / mixed-escape policies

## 2026-03-28 20:25 UTC - Longer-prompt StateCache checks are mostly clean, with a narrow greedy drift at exact 256

I validated the same `8-bit` readout-only DeltaNet StateCache lane beyond exact `64` and added a teacher-forced surface for the same recurrent-state replay path.

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py benchmarks/bench_qwen35_deltanet_statecache_readout.py benchmarks/bench_qwen35_deltanet_statecache_loss.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state or statecache_readout or statecache_loss'`
  - result: `7 passed`

Teacher-forced CUDA check on `Qwen/Qwen3.5-0.8B`, `prefix_length=256`, `eval_steps=16`:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_loss.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --sequence-length 272 --prefix-length 256 --eval-steps 16 --group-size 32 --bits 8`
- `dense_teacher_forced_loss = 0.0764`
- `deltanet_statecache_teacher_forced_loss = 0.0852`
- `teacher_forced_loss_delta = 0.0088`
- `dense_teacher_forced_perplexity = 1.0794`
- `deltanet_statecache_teacher_forced_perplexity = 1.0890`
- `teacher_forced_perplexity_ratio = 1.0088`
- `teacher_forced_token_agreement_rate = 1.0`
- `deltanet_statecache_teacher_forced_target_match_rate = 1.0`
- runtime is close to dense on this surface:
  - `dense_decode_ms_per_step = 21.16`
  - `deltanet_statecache_decode_ms_per_step = 22.76`

Generated-token CUDA checks:

- exact `256`, `4` generated tokens:
  - command:
    - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 256 --max-new-tokens 4 --bits 8 --continue-on-error`
  - dense ids:
    - `[4778, 45543, 13, 7976]`
  - StateCache ids:
    - `[4778, 45543, 13, 198]`
  - `deltanet_statecache_greedy_token_agreement_rate = 0.75`
  - `deltanet_statecache_output_max_abs_error = 0.0084`
  - `deltanet_statecache_max_abs_error = 0.0256`
  - runtime:
    - `dense_decode_ms_per_step = 115.61`
    - `deltanet_statecache_decode_ms_per_step = 38.50`

- exact `512`, `4` generated tokens:
  - command:
    - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 512 --max-new-tokens 4 --bits 8 --continue-on-error`
  - dense ids:
    - `[65789, 12482, 364, 4778]`
  - StateCache ids:
    - `[65789, 12482, 364, 4778]`
  - `deltanet_statecache_greedy_token_agreement_rate = 1.0`
  - `deltanet_statecache_output_max_abs_error = 0.0057`
  - `deltanet_statecache_max_abs_error = 0.0255`
  - runtime:
    - `dense_decode_ms_per_step = 129.33`
    - `deltanet_statecache_decode_ms_per_step = 37.58`

So the current picture is:

- the readout-only `8-bit` StateCache lane is still numerically tight at longer prefixes
- teacher-forced quality at `256` is essentially flat
- exact generation is not uniformly perfect yet, but the observed failure is narrow: a single-token divergence at exact `256`, while exact `512` still matches fully

That makes the next StateCache step narrower:

- treat this as a greedy-stability issue, not a broad recurrent-state corruption issue
- inspect the exact `256` divergence step rather than redesigning the whole `8-bit` replay path

## 2026-03-28 20:05 UTC - Captured DeltaNet StateCache sweeps now bridge real Qwen3.5 state into the simulator

I added a small bridge from the dense Qwen3.5 DeltaNet inspection lane into the StateCache simulator:

- the inspect runner can now save a real recurrent or conv state sample as `.npz`
- the simulator can consume that captured sample directly
- the new `bench_qwen35_statecache_real_sweep.py` wrapper can capture and summarize early/mid/late layer sweeps

Useful first real-state read from this Mac:

- captured recurrent state
- layer `0`
- prompt length `7`
- decode steps `1`

Results:

- `M0 8b`
  - compression ratio: `3.2x`
  - final update error: `0.0047`
  - final readout error: `0.0255`
- `M0 4b`
  - compression ratio: `5.33x`
  - final update error: `0.0592`
  - final readout error: `0.4508`
- `M0 3b`
  - compression ratio: `6.4x`
  - final update error: `0.1214`
  - final readout error: `0.7355`

Renorm did not change that first sample because it only contained one decode step. That is still useful: the bridge is working, and the next informative local sweep is multi-step captured recurrent state on early/mid/late layers.

## 2026-03-28 22:35 UTC - Multi-step recurrent-state sweep shows layer-dependent renorm behavior

I ran the new real-state sweep wrapper on Qwen3.5 recurrent state with:

- prompt length `32`
- decode steps `4`
- layers `0`, `12`, `22`

The useful read is:

- early layer `0`
  - `8b`: update `0.0056`, readout `0.0435`
  - `4b`: update `0.1019`, readout `0.7565`
  - `3b`: update `0.2061`, readout `1.3190`
  - best renorm interval stayed `0`
- mid layer `12`
  - `8b`: update `0.0091`, readout `0.0746`
  - `4b`: update `0.0887`, readout `1.0221`
  - `3b`: update `0.1193`, readout `1.5740`
  - best renorm interval was `2` for all `M0` bitwidths
- late layer `22`
  - `8b`: update `0.0162`, readout `0.1128`
  - `4b`: update `0.1514`, readout `1.1857`
  - `3b`: update `0.3026`, readout `3.9954`
  - best renorm interval stayed `0`

So the current local StateCache hint for the CUDA box is:

- `M0 8b` looks like the first serious recurrent-state candidate
- `M0 4b` is already quite marginal on readout by the time we hit late layers
- `M0 3b` is too lossy on this captured recurrent-state slice
- renorm is not a universal win
  - it helped the mid layer
  - it did not help the early or late layer on this `4`-step slice

That suggests the CUDA implementation should probably start with:

- recurrent-state `8b`
- optional renorm, but measured per layer family rather than assumed globally

## 2026-03-28 22:55 UTC - Readout harness needed a fresh prefill before StateCache decode

The exact-`256` CUDA readout miss turned out to be a harness issue, not a new StateCache quality boundary.

Root cause:

- `run_qwen35_deltanet_statecache_readout_harness(...)` was starting StateCache generation from the same model execution used for dense DeltaNet capture
- the native Qwen3.5 linear-attention stack is not behaving as a purely stateless function of `past_key_values` across that extra capture path
- teacher-forced loss stayed clean because the loss harness already does a fresh prefill before its replay loop

Fix:

- the readout harness now runs a fresh dense prefill for the StateCache generation path
- the ablation summary is still reported, but it is no longer part of the benchmarked decode setup

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py benchmarks/bench_qwen35_deltanet_statecache_readout.py benchmarks/bench_qwen35_deltanet_statecache_loss.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state or statecache_readout or statecache_loss'`
  - result: `8 passed`

Recheck on `Qwen/Qwen3.5-0.8B`, exact `256`, `4` generated tokens:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 256 --max-new-tokens 4 --bits 8 --continue-on-error`
- dense ids:
  - `[4778, 45543, 13, 7976]`
- StateCache ids:
  - `[4778, 45543, 13, 7976]`
- `deltanet_statecache_greedy_token_agreement_rate = 1.0`
- `deltanet_statecache_output_max_abs_error = 0.0084`
- `deltanet_statecache_max_abs_error = 0.0256`

So the narrow exact-`256` miss is gone. The current `8-bit` readout-only lane is back to:

- exact `64`: clean
- exact `256`: clean
- exact `512`: clean
- teacher-forced `256`: clean

## 2026-03-28 23:20 UTC - First real resident recurrent-state lane is keepable on CUDA

I extended the Qwen3.5 DeltaNet StateCache harnesses so they can run two explicit runtime stages:

- `readout_only_m0`
- `post_update_m0`

The new path also supports an optional `renorm_interval`, but the first CUDA checks show that renorm should remain off by default for now.

Code surface:

- [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py)
  - `Qwen35DeltaNetStateHarness.run_deltanet_statecache_readout(...)`
  - `Qwen35DeltaNetStateHarness.evaluate_deltanet_statecache_loss(...)`
  - `run_qwen35_deltanet_statecache_readout_harness(...)`
  - `run_qwen35_deltanet_statecache_loss_harness(...)`
- [bench_qwen35_deltanet_statecache_readout.py](/workspace/DotCache/benchmarks/bench_qwen35_deltanet_statecache_readout.py)
- [bench_qwen35_deltanet_statecache_loss.py](/workspace/DotCache/benchmarks/bench_qwen35_deltanet_statecache_loss.py)

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py benchmarks/bench_qwen35_deltanet_statecache_readout.py benchmarks/bench_qwen35_deltanet_statecache_loss.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state or statecache_readout or statecache_loss'`
  - result: `9 passed`

Exact `64`, `4` generated tokens:

- `readout_only_m0`, `8b`, no renorm:
  - agreement `1.0`
  - output max abs error `0.0060`
  - state max abs error `0.0256`
- `post_update_m0`, `8b`, no renorm:
  - agreement `1.0`
  - output max abs error `0.0089`
  - state max abs error `0.0396`
- `post_update_m0`, `8b`, `renorm_interval=2`:
  - agreement `0.75`
  - last token diverged

Teacher-forced `256` (`prefix_length=256`, `eval_steps=16`):

- `readout_only_m0`, `8b`, no renorm:
  - `teacher_forced_loss_delta = 0.0088`
  - `teacher_forced_perplexity_ratio = 1.0088`
  - token agreement `1.0`
- `post_update_m0`, `8b`, no renorm:
  - `teacher_forced_loss_delta = 0.0088`
  - `teacher_forced_perplexity_ratio = 1.0088`
  - token agreement `1.0`

Exact generation with the real resident recurrent-state path:

- exact `256`, `4` generated tokens, `post_update_m0`, `8b`, no renorm:
  - dense ids `[4778, 45543, 13, 7976]`
  - StateCache ids `[4778, 45543, 13, 7976]`
  - agreement `1.0`
  - output max abs error `0.0095`
  - state max abs error `0.0492`
- exact `512`, `4` generated tokens, `post_update_m0`, `8b`, no renorm:
  - dense ids `[65789, 12482, 364, 4778]`
  - StateCache ids `[65789, 12482, 364, 4778]`
  - agreement `1.0`
  - output max abs error `0.0107`
  - state max abs error `0.0547`

So the current DeltaNet StateCache conclusion is:

- `post_update_m0` at `8-bit` is the first keepable resident recurrent-state runtime on CUDA
- it holds on exact `64`, `256`, and `512`, and on teacher-forced `256`
- `renorm_interval=2` is already harmful on the short exact lane, so renorm should stay off by default until we have a better layer-aware policy

## 2026-03-28 23:35 UTC - `post_update_m0` also holds at exact 1024 and teacher-forced 1024

I pushed the first resident recurrent-state lane to the next obvious boundary before treating it as more than a prototype.

Exact `1024`, `4` generated tokens, `post_update_m0`, `8b`, no renorm:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 1024 --max-new-tokens 4 --bits 8 --state-stage post_update_m0 --renorm-interval 0 --continue-on-error`
- dense ids:
  - `[12482, 364, 4778, 45543]`
- StateCache ids:
  - `[12482, 364, 4778, 45543]`
- `deltanet_statecache_greedy_token_agreement_rate = 1.0`
- `deltanet_statecache_output_max_abs_error = 0.0098`
- `deltanet_statecache_max_abs_error = 0.0505`
- runtime:
  - `dense_decode_ms_per_step = 136.97`
  - `deltanet_statecache_decode_ms_per_step = 16.47`

Teacher-forced `1024` (`prefix_length=1024`, `eval_steps=32`), `post_update_m0`, `8b`, no renorm:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_loss.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --sequence-length 1056 --prefix-length 1024 --eval-steps 32 --group-size 32 --bits 8 --state-stage post_update_m0 --renorm-interval 0`
- `teacher_forced_loss_delta = 0.0003`
- `teacher_forced_perplexity_ratio = 1.0003`
- `teacher_forced_token_agreement_rate = 1.0`
- `deltanet_statecache_teacher_forced_target_match_rate = 1.0`
- recurrent compression ratio stayed `3.2x`

So the current productization read is stronger now:

- `post_update_m0` at `8-bit` is not just a short-context curiosity
- it stays clean through exact `1024` and teacher-forced `1024`
- this is strong enough to treat as the default CUDA StateCache candidate, with renorm still off by default

## 2026-03-28 23:55 UTC - Productized Qwen3.5 CUDA StateCache lane and compared it against the other native paths

I promoted the `Qwen/Qwen3.5-0.8B` CUDA StateCache path into the normal repo surfaces instead of leaving it as a prototype-only bench.

Surface changes:

- [scripts/run_qwen35_0p8b_statecache_cuda.sh](/workspace/DotCache/scripts/run_qwen35_0p8b_statecache_cuda.sh)
  - first-class CUDA runner for the default StateCache lane
- [bench_model_matrix.py](/workspace/DotCache/benchmarks/bench_model_matrix.py)
  - `qwen35_0p8b_hf` now emits the StateCache runner on `torch_cuda`
  - it still emits the dense text runner on non-CUDA backends
- [model_registry.py](/workspace/DotCache/dotcache/model_registry.py)
  - updated the `qwen35_0p8b_hf` note to reflect the current recommended CUDA lane

Shared exact `64`, `4` generated tokens, CUDA comparison:

- dense text lane:
  - command:
    - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_text.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 64 --max-new-tokens 4 --continue-on-error`
  - `dense_decode_ms_per_step = 57.26`
- attention-subset DotCache lane with the CUDA third-pass profile:
  - command:
    - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml --repeat-counts --target-prompt-lengths 64 --max-new-tokens 4 --continue-on-error`
  - `dotcache_decode_ms_per_step = 77.80`
  - `teacher_forced_logit_max_abs_error = 2.2676`
- DeltaNet StateCache lane (`post_update_m0`, `8b`, no renorm):
  - command:
    - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --repeat-counts --target-prompt-lengths 64 --max-new-tokens 4 --bits 8 --state-stage post_update_m0 --renorm-interval 0 --continue-on-error`
  - `deltanet_statecache_decode_ms_per_step = 17.09`
  - `deltanet_statecache_greedy_token_agreement_rate = 1.0`
  - `deltanet_statecache_output_max_abs_error = 0.0089`
  - `deltanet_statecache_effective_recurrent_compression_ratio = 3.2`

So the current Qwen3.5 CUDA read is:

- dense remains the accuracy/performance ceiling
- attention-subset DotCache is still the exploratory KV-compression lane and remains fidelity-limited
- DeltaNet StateCache `post_update_m0` is now the first productizable compressed native lane because it stays very close to dense while cutting resident recurrent state materially

## 2026-03-29 01:25 UTC - CUDA matrix refresh confirms the Qwen3.5 StateCache lanes

I reran the shared matrix surface only for the two Qwen3.5 CUDA entries that now route through DeltaNet StateCache:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_model_matrix.py --model-keys qwen35_0p8b_hf qwen35_4b_hf --run-supported --backend torch_cuda --device cuda --max-new-tokens 2 --output-format jsonl`

That refresh keeps the current StateCache defaults intact:

- `qwen35_0p8b_hf`
  - `post_update_m0`
  - `8-bit`
  - `renorm_interval = 0`
  - no recurrent escapes
- `qwen35_4b_hf`
  - `post_update_m0`
  - `8-bit`
  - `renorm_interval = 0`
  - recurrent `M3` escapes on layers `0`, `1`, and `2`

Exact-length CUDA results from the matrix rerun:

- `Qwen3.5 0.8B`
  - `512`: dense `42.49 ms/step`, StateCache `16.47 ms/step`, agreement `1.0`
  - `1024`: dense `41.12 ms/step`, StateCache `16.22 ms/step`, agreement `1.0`
  - fixed-resident bytes: `19,759,104 -> 6,782,976` (`2.91x` compression)
  - recurrent compression: `3.2x`
- `Qwen3.5 4B`
  - `512`: dense `80.22 ms/step`, StateCache `22.81 ms/step`, agreement `1.0`
  - `1024`: dense `84.05 ms/step`, StateCache `26.71 ms/step`, agreement `1.0`
  - fixed-resident bytes: `51,904,512 -> 21,626,880` (`2.4x` compression)
  - recurrent compression: `2.51x`

The repo-level conclusion is straightforward:

- both Qwen3.5 CUDA StateCache lanes are stable on the shared matrix surface
- `0.8B` remains the cleaner and more aggressive lane
- `4B` is viable as long as the early recurrent `M3` escapes stay in place
- the next Qwen3.5 product work should stay on StateCache or broader model-scale validation, not on productizing the combined hybrid-compression lane

## 2026-03-29 02:12 UTC - Longer-prompt CUDA scaling holds for the Qwen3.5 StateCache lanes

I extended the same CUDA matrix surface to longer exact prompts for the two Qwen3.5 StateCache entries:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_model_matrix.py --model-keys qwen35_0p8b_hf qwen35_4b_hf --run-supported --backend torch_cuda --device cuda --max-new-tokens 2 --prompt-lengths 2048 4096 --output-format jsonl`

The result is that the StateCache-only lanes stay stable at `2048` and `4096` for both currently supported Qwen3.5 models. All four exact-length runs kept `greedy_token_agreement_rate = 1.0`.

- `Qwen3.5 0.8B`
  - `2048`: dense `43.93 ms/step`, StateCache `16.67 ms/step`, `2.63x` faster
  - `4096`: dense `35.21 ms/step`, StateCache `16.65 ms/step`, `2.12x` faster
  - total tracked state bytes:
    - `2048`: `44,949,504 -> 31,973,376`, saving `12,976,128` bytes (`28.87%`)
    - `4096`: `70,115,328 -> 57,139,200`, saving `12,976,128` bytes (`18.51%`)
  - fixed-resident compression: `2.91x`
  - recurrent compression: `3.2x`
- `Qwen3.5 4B`
  - `2048`: dense `55.75 ms/step`, StateCache `23.00 ms/step`, `2.42x` faster
  - `4096`: dense `88.41 ms/step`, StateCache `23.72 ms/step`, `3.73x` faster
  - total tracked state bytes:
    - `2048`: `119,078,912 -> 88,801,280`, saving `30,277,632` bytes (`25.43%`)
    - `4096`: `186,187,776 -> 155,910,144`, saving `30,277,632` bytes (`16.26%`)
  - fixed-resident compression: `2.4x`
  - recurrent compression: `2.51x`

The main pattern is exactly what the smaller matrix already suggested:

- absolute savings stay flat with prompt length because StateCache is compressing fixed resident recurrent state, not the token-growing attention state
- total saving percentage falls as the prompt gets longer because the uncompressed token-growing portion becomes a larger share of total state
- decode speedups still hold well at longer context on both models, so this remains a real product lane rather than a short-prompt artifact

I then extended the same exact-length ladder again on this pod:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_model_matrix.py --model-keys qwen35_0p8b_hf qwen35_4b_hf --run-supported --backend torch_cuda --device cuda --max-new-tokens 2 --prompt-lengths 8192 16384 32768 65536 --output-format jsonl`

That establishes the current tested StateCache ceiling here:

- both Qwen3.5 StateCache lanes are still clean at exact `8192` and `16384`
- both hit `OutOfMemoryError` at exact `32768` and `65536`
- the models still advertise `262144` max positions, so this is a pod/runtime ceiling rather than an architectural one

Extended exact-length CUDA results:

- `Qwen3.5 0.8B`
  - `8192`: dense `50.32 ms/step`, StateCache `17.09 ms/step`, `2.94x` faster, agreement `1.0`
  - `16384`: dense `27.77 ms/step`, StateCache `16.83 ms/step`, `1.65x` faster, agreement `1.0`
  - total tracked state savings:
    - `8192`: `10.77%`
    - `16384`: `5.87%`
- `Qwen3.5 4B`
  - `8192`: dense `83.16 ms/step`, StateCache `23.44 ms/step`, `3.55x` faster, agreement `1.0`
  - `16384`: dense `79.91 ms/step`, StateCache `26.54 ms/step`, `3.01x` faster, agreement `1.0`
  - total tracked state savings:
    - `8192`: `9.45%`
    - `16384`: `5.14%`

So the longer-context read is consistent with the shorter one:

- StateCache remains a real decode-speed win through exact `16384`
- total saving percentage keeps falling as token-growing attention state dominates the total
- on this pod, exact `16384` is the current reliable tested ceiling for both currently supported Qwen3.5 StateCache lanes

## 2026-03-28 23:59 UTC - First combined Qwen3.5 0.8B CUDA hybrid lane is runnable, but still exploratory

I added a new combined bench surface in [bench_qwen35_attention_subset_statecache_dotcache.py](/workspace/DotCache/benchmarks/bench_qwen35_attention_subset_statecache_dotcache.py) and a matching integration path in [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py) that does both:

- DotCache on the six `full_attention` layers `[3, 7, 11, 15, 19, 23]`
- DeltaNet StateCache on the eighteen `linear_attention` layers

The combined runtime uses the existing native hybrid carrier:

- full-attention KV is loaded into the attention-subset DotCache runtime
- linear-attention recurrent state is quantized with the same `post_update_m0`, `8-bit`, `renorm=0` StateCache path

Validation:

- `python -m py_compile benchmarks/bench_qwen35_attention_subset_statecache_dotcache.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'statecache_dotcache or attention_subset_statecache_dotcache or attention_subset_dotcache_harness'`
  - result: `5 passed`

Live CUDA smoke on `Qwen/Qwen3.5-0.8B`, exact `64`, `2` generated tokens:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_attention_subset_statecache_dotcache.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --torch-dtype float16 --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml --state-stage post_update_m0 --state-bits 8 --state-renorm-interval 0 --target-prompt-lengths 64 --max-new-tokens 2 --repeat-counts 1 --continue-on-error`
- result:
  - `dense_decode_ms_per_step = 32.80`
  - `dotcache_decode_ms_per_step = 47.88`
  - `teacher_forced_logit_max_abs_error = 0.6875`
  - `replay_context_max_abs_error = 0.2079`
  - `replay_output_max_abs_error = 0.0306`
  - `deltanet_statecache_effective_recurrent_compression_ratio = 3.2`
  - `deltanet_statecache_effective_fixed_resident_compression_ratio = 2.91`
  - `dotcache_decode_runtime_ms_total = 54.96`
  - `dotcache_append_runtime_ms_total = 1.40`
  - `dotcache_qkv_projection_ms_total = 4.30`
  - `dotcache_output_projection_ms_total = 0.53`

So the first true hybrid-compressed Qwen3.5 lane now exists and runs end to end on CUDA. The engineering read is:

- the DeltaNet StateCache half remains solid
- the combined lane is numerically promising enough to keep as an experimental surface
- it is not productizable yet because the full-attention DotCache decode half still dominates latency and leaves the hybrid path slower than dense at exact `64`

That changes the next optimization target again:

- do not tune the StateCache half first
- attack the full-attention DotCache decode cost inside the combined Qwen3.5 lane

I also tested the most obvious structural shortcut on that full-attention side: route the grouped prepared decode through the existing `output_only` backend path so it does not materialize logits and weights.

- code path:
  - temporary local change in [model_kv_cache.py](/workspace/DotCache/dotcache/model_kv_cache.py) and [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py)
- validation:
  - `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_torch_cuda_backend.py -k 'grouped_prepared_cuda_output_only_matches_full_decode'`
  - `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'statecache_dotcache or attention_subset_statecache_dotcache or attention_subset_dotcache_harness'`

That path was correctness-clean but not a speed win on the real combined exact-`64` lane:

- kept full grouped decode:
  - `dense_decode_ms_per_step = 32.80`
  - `dotcache_decode_ms_per_step = 47.88`
  - `dotcache_decode_runtime_ms_total = 54.96`
  - `teacher_forced_logit_max_abs_error = 0.6875`
- temporary grouped `output_only` route:
  - `dense_decode_ms_per_step = 35.47`
  - `dotcache_decode_ms_per_step = 48.59`
  - `dotcache_decode_runtime_ms_total = 57.16`
  - `teacher_forced_logit_max_abs_error = 0.6836`

So I backed that change out. The current conclusion is narrower:

- grouped `output_only` is not the next win for the Qwen3.5 combined lane
- the remaining latency problem is deeper inside the full-attention DotCache decode path than just logits/weights materialization

The next direct check was whether the grouped batched decode itself was the wrong shape for this workload. Qwen3.5 `0.8B` full-attention layers on this lane are a very small fixed CUDA case:

- `8` query heads
- `2` KV heads
- `head_dim = 256`
- `tokens_per_page = 16`
- exact `64` means only `4` pages per KV head

I forced the ungrouped per-KV-head fallback in a one-off run and it beat the grouped path on the real combined exact-`64` lane. I then kept that as a narrow adapter-level CUDA specialization in [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py), using a new `prefer_grouped_batching` switch in [model_kv_cache.py](/workspace/DotCache/dotcache/model_kv_cache.py).

Validation:

- `python -m py_compile dotcache/model_kv_cache.py dotcache/integrations/qwen35.py benchmarks/bench_qwen35_attention_subset_statecache_dotcache.py tests/test_qwen35_integration.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'statecache_dotcache or attention_subset_statecache_dotcache or attention_subset_dotcache_harness'`
  - result: `5 passed`

Updated exact `64`, `2` generated tokens, CUDA combined-lane checkpoint:

- previous grouped path:
  - `dense_decode_ms_per_step = 32.80`
  - `dotcache_decode_ms_per_step = 47.88`
  - `dotcache_decode_runtime_ms_total = 54.96`
  - `prepared_chunk_resident_bytes = 1,013,760`
  - `resident_bytes = 1,688,576`
  - `teacher_forced_logit_max_abs_error = 0.6875`
- kept ungrouped CUDA specialization:
  - `dense_decode_ms_per_step = 35.07`
  - `dotcache_decode_ms_per_step = 43.80`
  - `dotcache_decode_runtime_ms_total = 48.03`
  - `prepared_chunk_resident_bytes = 831,488`
  - `resident_bytes = 1,506,304`
  - `teacher_forced_logit_max_abs_error = 0.6797`

So the current Qwen3.5 combined-lane read is tighter now:

- the first keepable full-attention decode win on this hybrid lane is not a new kernel
- it is a workload-shaped dispatch choice: ungrouped per-KV-head decode is better than grouped batching for this small CUDA case
- the lane is still slower than dense at exact `64`, but the gap is smaller and the resident bytes also improved

The same dispatch choice also helps the standalone attention-subset-only Qwen3.5 lane, which confirms this is a full-attention workload-shape issue rather than something specific to the combined StateCache runtime.

Validation:

- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'attention_subset_dotcache_harness and not statecache'`
  - result: `3 passed`

Standalone attention-subset CUDA rerun on `Qwen/Qwen3.5-0.8B`, exact `64`, `2` generated tokens:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml --target-prompt-lengths 64 --max-new-tokens 2 --repeat-counts 1 --continue-on-error`
- result:
  - `dense_decode_ms_per_step = 38.33`
  - `dotcache_decode_ms_per_step = 42.12`
  - `dotcache_decode_runtime_ms_total = 43.91`
  - `prepared_chunk_resident_bytes = 831,488`
  - `resident_bytes = 1,506,304`
  - `teacher_forced_logit_max_abs_error = 0.7188`

That makes the current CUDA read clearer:

- Qwen3.5 full-attention layers want the per-KV-head fallback, not grouped batching, on this small `8q / 2kv / 4 pages-per-kv-head` shape
- this helps both the standalone attention-subset lane and the combined hybrid-compressed lane
- the next optimization target remains inside the full-attention DotCache decode math itself, but the dispatch policy is now better matched to the actual workload

## 2026-03-28 19:10 UTC - Qwen3.5 DeltaNet probes now run on CUDA and point at an 8-bit StateCache path

I fixed two real integration issues in the DeltaNet state-capture path in [qwen35.py](/workspace/DotCache/dotcache/integrations/qwen35.py):

- `CaptureQwen35DeltaNet` now retries the underlying `Qwen3_5GatedDeltaNet` call without `cache_position` when the installed Transformers build does not accept that kwarg.
- `Qwen35DeltaNetStateModelAdapter.__post_init__` now runs the shared Qwen3.5 linear-attention runtime configuration before installing the capture wrappers, so CPU/tiny test models still downgrade off the CUDA-only fast path cleanly.
- the ablation replay helper now uses the same compatibility shim instead of calling the raw DeltaNet module directly.

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py tests/test_qwen35_integration.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state'`
  - result: `4 passed`

Live CUDA DeltaNet inspection on `Qwen/Qwen3.5-0.8B`:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_state_inspect.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --target-prompt-lengths 64 256 --max-new-tokens 4 --continue-on-error`
- exact `64`:
  - `hybrid_state_total_bytes = 20,594,688`
  - `deltanet_conv_state_bytes = 884,736`
  - `deltanet_recurrent_state_bytes = 18,874,368`
  - `hybrid_token_growing_bytes = 835,584`
- exact `256`:
  - `hybrid_state_total_bytes = 22,953,984`
  - `hybrid_token_growing_bytes = 3,194,880`

That confirms the structural split:

- most hybrid bytes are fixed resident DeltaNet state
- only the six full-attention layers `[3, 7, 11, 15, 19, 23]` are token-growing

Live CUDA DeltaNet ablation on exact `64`:

- command:
  - `source scripts/env_cuda.sh && .venv/bin/python benchmarks/bench_qwen35_deltanet_state_ablation.py --model-id Qwen/Qwen3.5-0.8B --backend torch_cuda --device cuda --target-prompt-lengths 64 --max-new-tokens 4 --bits 8 4 --continue-on-error`
- `8-bit`:
  - `readout_only_m0`: `max_abs_error = 0.0256`, `output_max_abs_error = 0.0060`
  - `pre_update_m0`: `max_abs_error = 0.0625`, `output_max_abs_error = 0.0062`
  - `post_update_m0`: `max_abs_error = 0.0674`, `output_max_abs_error = 0.0088`
  - `full_state_path_m0`: `max_abs_error = 0.0712`, `output_max_abs_error = 0.0085`
- `4-bit`:
  - `readout_only_m0`: `max_abs_error = 0.4253`, `output_max_abs_error = 0.1289`
  - `pre_update_m0`: `max_abs_error = 0.9844`, `output_max_abs_error = 0.1318`
  - `post_update_m0`: `max_abs_error = 1.9896`, `output_max_abs_error = 0.1414`
  - `full_state_path_m0`: `max_abs_error = 1.9896`, `output_max_abs_error = 0.1318`

The useful conclusion is clear:

- a first StateCache experiment should target an `8-bit` DeltaNet state lane, not `4-bit`
- `readout_only_m0` is the safest first compression target
- `post_update` and full-path replay are materially more fragile than readout-only

## 2026-03-28 23:35 UTC - Combined conv and recurrent sweep points to `8b` first, with more renorm pressure on conv state

I extended the real-state sweep to run both DeltaNet state families on the same Qwen3.5 slice:

- prompt length `32`
- decode steps `4`
- layers `0`, `12`, `22`
- state kinds `recurrent` and `conv`

The combined recommendation output was simple and consistent:

- recurrent layers `0`, `12`, `22`
  - all three recommend `M0 8b`
  - renorm intervals: `0`, `2`, `0`
- conv layers `0`, `12`, `22`
  - all three also recommend `M0 8b`
  - renorm intervals: `2`, `2`, `2`

The important detail is that conv state is clearly noisier than recurrent state at the same bitwidths on this captured slice. Representative readout errors:

- recurrent, `8b`
  - layer `0`: `0.0435`
  - layer `12`: `0.0746`
  - layer `22`: `0.1128`
- conv, `8b`
  - layer `0`: `0.1314`
  - layer `12`: `0.0796`
  - layer `22`: `0.0878`

And the lower-bit picture stayed negative for both families:

- recurrent `4b`
  - readout drift ranged from `0.7565` to `1.1857`
- conv `4b`
  - readout drift ranged from `0.8630` to `1.2947`
- `3b` was materially worse still for both

So the current local StateCache handoff gets a little sharper:

- start with `8b` for both recurrent and conv state
- treat renorm as more important for conv state than for recurrent state
- do not start the CUDA path from `4b` or `3b` on either state family

## 2026-03-29 00:40 UTC - Short local selective `4b` recurrent overrides do not destabilize the `8b` resident lane

After merging the first CUDA StateCache win path, I added per-layer recurrent-state bit overrides to the resident Qwen3.5 StateCache readout/loss harnesses and probed a few small local override pockets on this Mac.

Exact `64`, `2` generated tokens, `post_update_m0`, default `8b`, no renorm:

- baseline `8b` everywhere
  - greedy agreement `1.0`
  - output max abs error `0.00534`
  - recurrent resident bytes `5,898,240`
  - recurrent compression ratio `3.20x`
- layer `12 -> 4b`
  - greedy agreement `1.0`
  - output max abs error `0.00534`
  - recurrent resident bytes `5,767,168`
  - recurrent compression ratio `3.27x`
- layer `22 -> 4b`
  - greedy agreement `1.0`
  - output max abs error `0.00534`
  - recurrent resident bytes `5,767,168`
  - recurrent compression ratio `3.27x`
- layers `12,22 -> 4b`
  - greedy agreement `1.0`
  - output max abs error `0.00534`
  - recurrent resident bytes `5,636,096`
  - recurrent compression ratio `3.35x`

So the honest local read is:

- small selective `4b` recurrent pockets are compatible with the current `8b` resident path on this short exact slice
- the local exact `64` probe is not yet long or hard enough to expose a failure boundary
- the next useful selective probe should move to a longer exact or teacher-forced slice rather than adding more short `64 / 2` cases

## 2026-03-29 01:10 UTC - Longer local teacher-forced probe shows selective `4b` is plausible but no longer free

I followed the short exact-`64` probe with a longer teacher-forced check on the merged StateCache lane:

- `sequence_length = 160`
- `prefix_length = 128`
- `eval_steps = 16`
- `state_stage = post_update_m0`
- baseline `8b` everywhere vs selective recurrent override `12:4 22:4`

Baseline `8b` result:

- `teacher_forced_loss_delta = -0.00088`
- `teacher_forced_perplexity_ratio = 0.99912`
- recurrent resident bytes `5,898,240`

Selective `12:4 22:4` result:

- `teacher_forced_loss_delta = +0.00327`
- `teacher_forced_perplexity_ratio = 1.00327`
- recurrent resident bytes `5,636,096`

So the current local read tightens up:

- a small two-layer `4b` recurrent pocket still looks plausible on top of the `8b` resident lane
- but at a longer prefix it is no longer perfectly free
- the cost is small enough to keep exploring, yet real enough that the next step should be layer-by-layer on CUDA rather than assuming every short local success transfers unchanged

Live CUDA check:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --target-prompt-lengths 64 \
  --max-new-tokens 2
```

Useful runtime fields on that run:

- `native_hybrid_fixed_resident_layer_ids`:
  - `[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]`
- `native_hybrid_token_growing_layer_ids`:
  - `[3, 7, 11, 15, 19, 23]`
- `native_hybrid_prefill_fixed_resident_bytes`:
  - `19,759,104`
- `native_hybrid_final_fixed_resident_bytes`:
  - `19,759,104`
- `native_hybrid_fixed_resident_growth_bytes`:
  - `0`
- `native_hybrid_prefill_token_growing_bytes`:
  - `0`
- `native_hybrid_final_token_growing_bytes`:
  - `0`

The fidelity result stayed on the same CUDA third-pass checkpoint:

- replay context max abs error: `0.2076`
- replay output max abs error: `0.0314`
- teacher-forced logit max abs error: `0.7148`

So the next implementation step is no longer “discover the split.” It is to exploit it: build any later Qwen3.5 hybrid caching around a resident linear-attention state object plus the existing attention-subset DotCache KV path.

## 2026-03-28 17:50 UTC - Qwen3.5 DotCache runtime now owns a native hybrid state object

I turned the earlier split reporting into a real runtime object carried by the Qwen3.5 attention-subset adapter.

The adapter now owns a `Qwen35NativeHybridRuntimeState` that:

- captures the post-handoff native `past_key_values` state after attention-subset KV has been replaced by placeholders
- preserves the fixed resident linear-attention partition
- refreshes the current native hybrid partition after each decode step
- emits the final runtime summary from that state object instead of recomputing an isolated one-shot partition at the end

That is the first keepable piece of actual hybrid-state machinery, not just instrumentation.

Validation:

- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'dotcache_harness or hybrid_state'`
  - result: `6 passed`

Live CUDA rerun on the third-pass profile stayed on the same fidelity checkpoint:

- replay context max abs error: `0.2076`
- replay output max abs error: `0.0314`
- teacher-forced logit max abs error: `0.7148`

And the runtime-owned native state object reported the expected invariant:

- `native_hybrid_fixed_resident_preserved = true`
- `native_hybrid_fixed_resident_growth_bytes = 0`

So the next Qwen3.5 step is narrower again:

- stop just carrying the native resident state object
- start using it to define a true hybrid cache interface for generation and benchmarking

## 2026-03-28 18:05 UTC - Qwen3.5 attention-subset runtime now has a single hybrid runtime object

I wrapped the native resident state and DotCache KV state into one runtime-owned object:

- `Qwen35NativeHybridRuntimeState`
- `Qwen35HybridDotCacheRuntimeState`

The adapter now carries that combined runtime object across decode, refreshes the native side after each step, and emits both the native hybrid summary and the DotCache resident/page summaries from one place.

This is still not the final generic hybrid cache API, but it is the first coherent runtime surface for Qwen3.5 that spans:

- fixed resident linear-attention state
- token-growing full-attention DotCache KV state

Validation:

- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'dotcache_harness or hybrid_state'`
  - result: `6 passed`

## 2026-03-29 00:40 UTC - Serving-only StateCache extends the 4B long-context ceiling

I added a serving-style DeltaNet StateCache harness that removes the dense-vs-StateCache side-by-side comparison path and only runs:

- one dense prefill to obtain the native recurrent state
- recurrent-state compression into StateCache form
- StateCache-only decode

That matters for long-context scaling because the earlier `Qwen3.5-4B @ 32768` failures were coming from compare-mode peak VRAM, not the steady-state resident StateCache bytes.

New bench surface:

- `benchmarks/bench_qwen35_deltanet_statecache_serving.py`

New runtime mode:

- `runtime_mode = "statecache_serving_only"`

Validation:

- `python -m py_compile dotcache/integrations/qwen35.py benchmarks/bench_qwen35_deltanet_statecache_readout.py benchmarks/bench_qwen35_deltanet_statecache_serving.py tests/test_qwen35_integration.py`
- `PYTHONPATH=/workspace/DotCache .venv/bin/pytest -q tests/test_qwen35_integration.py -k 'deltanet_state or statecache_readout or statecache_serving'`
  - result: `15 passed`

The decisive run was:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_serving.py \
  --model-id Qwen/Qwen3.5-4B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 16384 32768 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --recurrent-mode-override layer:0=M3 \
  --recurrent-mode-override layer:1=M3 \
  --recurrent-mode-override layer:2=M3 \
  --continue-on-error
```

This answers the main scaling question:

- compare-mode `Qwen3.5-4B @ 32768` still OOMed, even with `bnb_8bit`
- serving-only `Qwen3.5-4B @ 32768` passes on the same pod

Serving-only checkpoints with `bnb_8bit` weights:

- exact `16384`
  - `deltanet_statecache_decode_ms_per_step = 97.46`
  - prefill peak allocated/reserved: `14.02 GB / 17.21 GB`
  - decode peak allocated/reserved: `14.55 GB / 17.21 GB`
- exact `32768`
  - `deltanet_statecache_decode_ms_per_step = 98.41`
  - prefill peak allocated/reserved: `22.77 GB / 23.43 GB`
  - decode peak allocated/reserved: `23.80 GB / 24.19 GB`

Resident-state accounting at `32768`:

- fixed resident dense bytes: `51.90 MB`
- fixed resident StateCache bytes: `21.63 MB`
- token-growing bytes: `1.0738 GB`

So the long-context blocker is now much clearer:

- StateCache itself scales further than the compare harness suggested
- the compare-mode ceiling was mostly benchmark overhead
- the true remaining limit is the token-growing full-attention half plus long-context prefill/runtime peak memory, not the compressed recurrent state

## 2026-03-29 01:05 UTC - Serving-only StateCache reaches 32768 on both tested Qwen3.5 members

I extended the serving-only StateCache ladder to both currently supported Qwen3.5 models with `bnb_8bit` weights and the same StateCache runtime:

- `state_stage = post_update_m0`
- `bits = 8`
- `renorm_interval = 0`
- `Qwen3.5-4B` keeps the known recurrent `M3` escapes on layers `0,1,2`

The useful result is that both models now reach exact `32768` on the serving-only path, while both still fail at exact `65536`.

Commands:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_serving.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 32768 65536 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --continue-on-error
```

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_serving.py \
  --model-id Qwen/Qwen3.5-4B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 32768 65536 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --recurrent-mode-override layer:0=M3 \
  --recurrent-mode-override layer:1=M3 \
  --recurrent-mode-override layer:2=M3 \
  --continue-on-error
```

Serving-only checkpoints:

- `Qwen3.5-0.8B @ 32768`
  - `deltanet_statecache_decode_ms_per_step = 74.60`
  - `13.41 tok/s`
  - prefill peak allocated/reserved: `17.91 GB / 20.33 GB`
  - decode peak allocated/reserved: `18.29 GB / 20.33 GB`
  - fixed resident saving: `65.67%`
  - total state saving: `3.07%`
- `Qwen3.5-4B @ 32768`
  - `deltanet_statecache_decode_ms_per_step = 100.86`
  - `9.91 tok/s`
  - prefill peak allocated/reserved: `22.77 GB / 29.11 GB`
  - decode peak allocated/reserved: `23.80 GB / 29.11 GB`
  - fixed resident saving: `58.33%`
  - total state saving: `2.69%`

Both `65536` runs still failed with `OutOfMemoryError`, and both failures were dominated by a single huge allocation request:

- attempted allocation: `30.31 GiB`

So the current serving-mode read is:

- serving-only StateCache is the right way to test long-context scaling
- the real ceiling on this pod is now exact `32768` for both tested Qwen3.5 members
- the remaining long-context limiter is not recurrent-state storage
- it is the token-growing full-attention half plus the large long-context runtime allocation spike that still appears at `65536`

## 2026-03-29 01:50 UTC - Qwen3.5-9B is viable on the same HF StateCache path

I pushed the next larger native HF model, `Qwen/Qwen3.5-9B`, through the same StateCache path.

Before the run, I cleaned the persistent cache so the model would fit without using a second temporary cache root:

- removed `/workspace/.cache/pip`
- removed duplicate HF GGUF snapshot for `bartowski/Llama-3.2-3B-Instruct-GGUF`
- removed the temporary `/tmp/hf-qwen35-9b-cache`

That left a single persistent copy of `Qwen3.5-9B` under `/workspace/.cache/huggingface`.

First, I ran a short exact-length feasibility pass with compare-mode StateCache and `bnb_8bit` weights:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 512 1024 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --continue-on-error
```

Feasibility results:

- exact `512`
  - greedy agreement `1.0`
  - `deltanet_statecache_output_max_abs_error = 0.03125`
  - dense decode `4.21 tok/s`
  - StateCache decode `10.27 tok/s`
  - speedup `2.44x`
- exact `1024`
  - greedy agreement `1.0`
  - `deltanet_statecache_output_max_abs_error = 0.046875`
  - dense decode `4.30 tok/s`
  - StateCache decode `10.32 tok/s`
  - speedup `2.40x`

Resident-state compression on `9B`:

- fixed resident dense bytes: `51.90 MB`
- fixed resident StateCache bytes: `17.30 MB`
- fixed resident saving: `66.67%`

Then I ran the serving-only long-context ladder:

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_serving.py \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 16384 32768 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --continue-on-error
```

Serving-only checkpoints:

- `Qwen3.5-9B @ 16384`
  - `deltanet_statecache_decode_ms_per_step = 99.89`
  - `10.01 tok/s`
  - prefill peak allocated/reserved: `20.33 GB / 24.02 GB`
  - decode peak allocated/reserved: `20.82 GB / 24.02 GB`
  - fixed resident saving: `66.67%`
  - total state saving: `5.88%`
- `Qwen3.5-9B @ 32768`
  - `deltanet_statecache_decode_ms_per_step = 103.75`
  - `9.64 tok/s`
  - prefill peak allocated/reserved: `29.14 GB / 29.32 GB`
  - decode peak allocated/reserved: `30.07 GB / 30.36 GB`
  - fixed resident saving: `66.67%`
  - total state saving: `3.07%`

So the next-larger-family read is positive:

- `Qwen3.5-9B` works on the same native HF StateCache path
- it reaches exact `32768` on the serving-only methodology
- it sits extremely close to the pod ceiling at that point
- the serving-only StateCache methodology now scales across `0.8B`, `4B`, and `9B`

## 2026-03-28 00:15 UTC - Qwen3.5 local runtime ablations now cover conv state as a first-class family

I extended the local Qwen3.5 DeltaNet StateCache debugging lane so it can ablate and localize conv state separately from recurrent state.

What landed:

- `statecache_scope = recurrent_only | conv_only | conv_plus_recurrent`
- matching conv-side runtime knobs:
  - `conv_bits`
  - `conv_layer_bits_overrides`
  - `conv_mode_overrides`
- conv-aware localization output for:
  - first recurrent failure layer
  - first conv failure layer
  - first combined DeltaNet failure layer
  - first combined-family failure kind in the hybrid localizer

This is intentionally still a local debugging surface, not a promoted CUDA default. The main value is that combined Qwen3.5 runs can now answer a cleaner question than before:

- does drift start on the attention subset
- on recurrent state
- on conv state
- or only when multiple compressed families interact

That should make the CUDA combined DotCache+StateCache bring-up much easier to debug, especially when a regression is small enough that a single scalar loss number is not very informative.

## 2026-03-28 20:20 UTC - First TurboQuant CUDA comparison slice: TinyLlama works, SmolLM2 exposes a Turbo3 fork limitation

I moved the external-runtime comparison from planning into the first real CUDA rows using the TurboQuant llama.cpp fork built at:

- `/workspace/llama-cpp-turboquant-cuda`

The comparison setup is:

- DotCache / HF baseline from `bench_llama_compare.py`
- external reference from raw `llama-cli` runs on the TurboQuant CUDA fork
- shared prompt family: `"Cache locality matters for fast decoding."` repeated `64` times

TinyLlama first-pass comparison (`repeat_count = 64`):

- HF dense baseline:
  - prompt length `577`
  - `dense_decode_ms_per_step = 8.73`
  - `114.51 tok/s`
- HF DotCache exact `M0/M0`:
  - `decode_ms_per_step = 91.81`
  - `10.89 tok/s`
  - greedy agreement `1.0`
- TurboQuant external `q8_0` KV:
  - prompt `17595.4 tok/s`
  - generation `336.1 tok/s`
- TurboQuant external `turbo3 uniform`:
  - prompt `17403.3 tok/s`
  - generation `275.3 tok/s`
- TurboQuant external `turbo3 LA-1`:
  - prompt `17974.1 tok/s`
  - generation `297.9 tok/s`

So the first honest external read is what we expected: the specialized llama.cpp CUDA KV-quant runtime is dramatically faster than the current HF DotCache TinyLlama lane, and still materially faster than the HF dense baseline.

SmolLM2 `360M` first-pass comparison is useful too, but mainly as a compatibility result:

- HF dense baseline:
  - prompt length `448`
  - `dense_decode_ms_per_step = 37.83`
  - `26.43 tok/s`
- HF DotCache exact `M0/M0`:
  - `decode_ms_per_step = 167.35`
  - `5.98 tok/s`
  - greedy agreement `1.0`
- TurboQuant external `q8_0` KV:
  - prompt `20526.5 tok/s`
  - generation `335.2 tok/s`
- TurboQuant external `turbo3 uniform`, `turbo3 LA-1`, and `turbo3 LA-5`:
  - all fail on the current CUDA fork with:
    - `GGML_ASSERT(ne00 % QK_TURBO3_GROUP == 0) failed`
    - source location `ggml/src/ggml-cuda/set-rows.cu:333`

So the current comparison story is:

- TinyLlama gives a clean apples-to-apples first slice across HF dense, HF DotCache, TurboQuant `q8_0`, and TurboQuant `turbo3`
- SmolLM2 already shows that this CUDA TurboQuant fork is not universally compatible yet; `q8_0` works, but Turbo3-mode KV quantization asserts on this model shape

That means the next comparison work should stay narrow:

- finish the JSONL/reporter path for the working TinyLlama slice
- treat SmolLM2 as a documented negative-compatibility data point until the external fork fixes the Turbo3 CUDA assert

## 2026-03-28 20:30 UTC - Llama 3.2 3B is the next clean Turbo3-compatible comparison model

I switched the external comparison set to Turbo3-compatible models only instead of trying to force SmolLM2 into the headline table.

The next local candidate from the persistent GGUF cache was:

- `/workspace/models/gguf/llama32_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf`

On the same `repeat_count = 64` prompt family, the raw TurboQuant CUDA runs are clean:

- external `q8_0`:
  - prompt `7929.7 tok/s`
  - generation `123.0 tok/s`
- external `turbo3 uniform`:
  - prompt `6206.2 tok/s`
  - generation `153.8 tok/s`
- external `turbo3 LA-1`:
  - prompt `10208.6 tok/s`
  - generation `205.7 tok/s`

The matching HF / DotCache baseline on `meta-llama/Llama-3.2-3B-Instruct` is:

- prompt length `449`
- HF dense:
  - `dense_decode_ms_per_step = 37.90`
  - `26.39 tok/s`
- HF DotCache exact `M0/M0`:
  - `decode_ms_per_step = 98.39`
  - `10.16 tok/s`
  - greedy agreement `1.0`

So the external comparison set should now be framed as:

- TinyLlama `1.1B`: first fully clean Turbo3 slice
- Llama `3.2 3B`: second clean Turbo3 slice, and a better “real model” comparison than TinyLlama alone
- SmolLM2 `360M`: compatibility-failure note for the current CUDA fork, not a primary Turbo3 benchmark

This is enough to make the external story credible without overstating support:

- Turbo3 works on at least two Llama-style models on this CUDA fork
- it is substantially faster than the current HF DotCache path on both
- model compatibility is still runtime-dependent, so the comparison section should explicitly separate “supported models” from “known failing models”
