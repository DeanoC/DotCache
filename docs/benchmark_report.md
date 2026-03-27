# DotCache MVP Results

This report is the compact benchmark summary for the current DotCache prototype. It sits above the experiment log in [performance_journal.md](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md) and the raw append-only records in [history.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/history.jsonl).

The prototype state reflected here is:

- exact-first `M0` affine plus `M3` escape execution
- compressed-domain `score_page` and `mix_page`
- standalone decode harnesses
- Phase 5 Llama-family model integration
- Phase 6 vLLM offline-adapter implementation
- `torch_mps` and `torch_cuda` accelerator backends

## Configuration

### Core page configuration

- group size: `32`
- bits_k: `4`
- bits_v: `4`
- tokens per page: usually `256` in the real-model path
- layouts: `group_major` for both K and V
- quantization: affine `M0` by default, `M3` escape for tails / high-precision paths
- execution mode in model benchmarks: exact full-context decode, batch `1`, greedy generation

### Benchmark environments

| Environment | Backend | Notes |
|---|---|---|
| Apple M4 Mac | `torch_mps` | Main optimization target so far |
| NVIDIA RTX 2000 Ada 16 GB | `torch_cuda` | First shared-torch parity port |

### Primary real-model checkpoints

| Model | Context Limit | Main Use |
|---|---:|---|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `2048` | Short-to-mid exact model integration validation |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | `8192` | Higher-context exact model benchmark on current machines |

## Correctness

- pack/unpack: covered by the unit suite and compressed-domain parity tests; current suite status is `85 passed, 5 skipped`
- score parity: standalone MPS and CUDA decode-step ladders stay in the low `1e-5` max-abs-logit range on recorded runs
- mix parity: standalone output errors stay in the low `1e-6` range on recorded exact ladders
- full attention parity: exact session and decode harnesses continue to match the explicit dequantized baseline within test tolerances
- end-to-end generation sanity:
  - tiny-random Llama smoke runs keep `1.0` greedy agreement on `cpu_ref`, `torch_mps`, and `torch_cuda`
  - real TinyLlama and SmolLM2 exact DotCache paths keep `1.0` greedy agreement on the recorded MPS and CUDA comparisons

One important caveat: teacher-forced model-logit drift is still non-zero on real checkpoints, especially on longer contexts, so the current end-to-end claim is "stable exact decode behavior with matching greedy tokens on tested prompts", not "bit-identical logits to dense attention."

### Teacher-forced quality snapshot

The new [bench_llama_loss.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_loss.py) harness is now the best local quality yardstick for `M1` and `M2`, because greedy agreement and raw max-logit drift were too coarse on short continuations.

| Model | Prefix / Eval | Mode | DotCache Loss Delta | Perplexity Ratio | Token Agreement | Max Abs Logit Drift |
|---|---:|---|---:|---:|---:|---:|
| TinyLlama | `288 / 32` | `K=M0, V=M0` | `-0.00101` | `0.99899` | `1.00` | `8.15` |
| TinyLlama | `288 / 32` | `K=M0, V=M1` | `-0.00014` | `0.99986` | `1.00` | `12.89` |
| TinyLlama | `288 / 32` | `K=M2, V=M0` | `+0.00002` | `1.00002` | `1.00` | `12.12` |
| TinyLlama | `288 / 32` | `K=M2 adaptive, V=M0` | `-0.00135` | `0.99865` | `1.00` | `8.70` |
| SmolLM2 360M | `1024 / 16` | `K=M0, V=M0` | `-0.00198` | `0.99803` | `1.00` | `7.57` |
| SmolLM2 360M | `1024 / 16` | `K=M0, V=M1` | `+0.01717` | `1.01732` | `1.00` | `14.54` |
| SmolLM2 360M | `1024 / 16` | `K=M2, V=M0` | `+0.02865` | `1.02907` | `1.00` | `17.64` |
| SmolLM2 360M | `1024 / 16` | `K=M2 adaptive, V=M0` | `+0.03948` | `1.04027` | `1.00` | `16.33` |

This changes the quality read in an important way:

- TinyLlama is relatively forgiving: `V`-only `M1` and adaptive `K`-only `M2` both keep teacher-forced loss almost flat on the tested continuation.
- SmolLM2 is not: both approximate modes materially worsen teacher-forced loss at `1024` prefix tokens, even though token agreement still stays at `1.0`.
- Adaptive segmented `M2` is now stable after decode bucketing, but it still regresses both quality and speed on SmolLM2 versus the fixed segmented variant, so it should remain experimental.
- That makes teacher-forced loss/perplexity the local quality metric to trust first for `M1/M2`, not greedy agreement alone.

Turbo3 now also has its own dedicated local MPS lane through [run_turbo3_mps_suite.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_turbo3_mps_suite.sh). The most important recent improvement there was a correct vectorized `3`-bit spill-unpack path on MPS using advanced indexing rather than repeated-index `torch.gather(...)`.

Latest local Turbo3 results:

| Model | Case | Dense Decode ms/step | Turbo3 Decode ms/step | KV Ratio | Agreement | Loss Delta |
|---|---|---:|---:|---:|---:|---:|
| TinyLlama | compare `10` | `396.83` | `1757.78` | `9.85x` | `1.00` | n/a |
| TinyLlama | compare `289` | `336.26` | `2631.47` | `0.55x` | `0.25` | n/a |
| TinyLlama | loss `288 / 32` | `450.86` | `3351.38` | n/a | `0.3125` | `+3.16766` |
| SmolLM2 360M | compare `7` | `667.43` | `5711.28` | `12.80x` | `1.00` | n/a |
| SmolLM2 360M | compare `1024` | `397.61` | `4306.74` | `0.27x` | `0.25` | n/a |
| SmolLM2 360M | loss `1024 / 16` | `544.12` | `5496.32` | n/a | `0.50` | `+2.45040` |

The most useful exact-prompt local comparison against existing DotCache modes is:

| Model | Prompt | Mode | Decode ms/step | Resident KV Bytes | Agreement | Max Abs Logit Error |
|---|---:|---|---:|---:|---:|---:|
| TinyLlama | `289` | `K=M0, V=M0` | `5730.80` | `7,929,856` | `1.00` | `0.5781` |
| TinyLlama | `289` | `K=M0, V=M1` | `4132.39` | `7,580,672` | `1.00` | `3.2510` |
| TinyLlama | `289` | `K=T3, V=T3` | `2631.47` | `7,208,960` | `0.25` | `26.6152` |
| SmolLM2 360M | `1024` | `K=M0, V=M0` | `5521.76` | `29,360,128` | `1.00` | `0.9769` |
| SmolLM2 360M | `1024` | `K=M0, V=M1` | `6224.08` | `26,820,608` | `1.00` | `3.8877` |
| SmolLM2 360M | `1024` | `K=T3, V=T3` | `4306.74` | `23,068,672` | `0.25` | `15.9824` |

That is the right read for now:

- Turbo3 is better optimized on MPS than it was before the vectorized spill-unpack work.
- It is still much worse on quality than `M0` or `V`-only `M1`.
- In these current exact reruns, Turbo3 can look faster than the noisier `M0` and `V`-only `M1` baselines, but only by paying unacceptable quality loss.
- So Turbo3 is still useful as a reference implementation and comparison point for future TurboQuant-style work, not a mode we should promote on the current MPS model path.

### One-load local mode profiling

The new [bench_llama_mode_profile.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_mode_profile.py) harness keeps one model loaded and reconfigures the local adapter across modes, which gives us a cleaner view of where time is going than the older compare runs.

TinyLlama exact `289` prompt, one loaded model:

| Mode | Decode ms/step | Score | Mix | Softmax | Unpack | FWHT | Prefill Ingest ms | Agreement | Max Abs Logit Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` | `5203.45` | `1503.95` | `1409.28` | `100.06` | `2196.55` | `0.00` | `900.88` | `1.00` | `0.5781` |
| `Turbo3` | `3917.59` | `882.65` | `687.37` | `105.42` | `802.08` | `30.89` | `8363.96` | `0.25` | `26.6152` |

SmolLM2 `1024` prompt, one loaded model:

| Mode | Decode ms/step | Score | Mix | Softmax | Unpack | FWHT | Prefill Ingest ms | Agreement | Max Abs Logit Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` | `9695.87` | `2686.13` | `2961.70` | `165.62` | `0.00` | `0.00` | `2271.62` | `1.00` | `0.9769` |
| `V-only M1` | `5409.58` | `1153.67` | `1163.36` | `153.08` | `648.66` | `0.00` | `10191.47` | `1.00` | `2.3926` |
| `Turbo3` | `8156.85` | `2711.92` | `2897.00` | `113.40` | `968.75` | `42.12` | `60617.45` | `0.25` | `15.9824` |

This profiling pass sharpened the local codec comparison:

- Turbo3 is no longer obviously losing on raw decode arithmetic. On TinyLlama `289`, it is lighter than `M0` in grouped score, grouped mix, and sampled unpack.
- That still does not make it a good local mode, because quality collapses and prefill ingest is far worse.
- `V`-only `M1` is the more useful DotCache-side comparison at SmolLM2 `1024`: it reduces grouped decode work and resident bytes while staying greedy-stable, but it still loses badly on prefill ingest and teacher-forced quality versus exact `M0`.
- On SmolLM2 `1024`, Turbo3 is also not a meaningful systems win: grouped score and mix stay close to `M0`, unpack remains heavy, and prefill ingest explodes.

## Memory

### Real-model KV-cache footprint

| Model / Prompt | Backend | Dense KV Bytes | DotCache KV Bytes | DotCache / Dense |
|---|---|---:|---:|---:|
| TinyLlama / `10` | MPS | `585,728` | `5,767,168` | `9.85x` |
| TinyLlama / `289` | MPS | `13,156,352` | `7,569,408` | `0.58x` |
| TinyLlama / `577` | MPS | `26,132,480` | `9,371,648` | `0.36x` |
| TinyLlama / `865` | MPS | `39,108,608` | `11,173,888` | `0.29x` |
| TinyLlama / `289` | CUDA | `13,156,352` | `7,569,408` | `0.58x` |
| TinyLlama / `577` | CUDA | `26,132,480` | `9,371,648` | `0.36x` |
| SmolLM2 / `2048` | MPS | `168,017,920` | `36,700,160` | `0.22x` |
| SmolLM2 / `2048` | CUDA | `168,017,920` | `36,700,160` | `0.22x` |
| SmolLM2 / `4096` | CUDA | `335,790,080` | `62,914,560` | `0.19x` |

### Memory takeaways

- DotCache loses badly on KV bytes for very short prompts because prepared tails dominate the footprint.
- The practical TinyLlama memory crossover on MPS happens between about `73` and `145` prompt tokens.
- On higher-context SmolLM2, DotCache is already near `22%` of dense KV bytes by `2048` tokens.
- The strongest current systems-value result is memory reduction, not universal decode-speed wins.

## Performance

### Standalone exact decode kernels

#### CUDA decode-step ladder

| Context | Prepare ms | Decode ms | Host-to-Device Bytes |
|---|---:|---:|---:|
| `64` | `7.22` | `2.31` | `10,240` |
| `256` | `0.35` | `3.10` | `40,960` |
| `1024` | `0.66` | `6.86` | `163,840` |
| `4096` | `2.38` | `21.50` | `655,360` |

These runs confirm the shared torch accelerator path is correct and scales cleanly, but they are still eager-kernel microbenchmarks rather than full serving wins.

### Exact model-path decode

#### TinyLlama on MPS

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Result |
|---|---:|---:|---|
| `145` | `36.46` | `172.30` | Dense faster |
| `289` | `44.30` | `254.40` | Dense faster |
| `577` | `79.12` | `279.30` | Dense faster |
| `865` | `171.99` | `282.56` | Dense faster |
| `1536` | `2274.82` | `4476.23` | Dense faster, but DotCache uses `0.24x` KV bytes |

#### TinyLlama on CUDA

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Result |
|---|---:|---:|---|
| `10` | `40.17` | `72.61` | Dense faster |
| `289` | `22.31` | `132.15` | Dense faster |
| `577` | `23.49` | `134.48` | Dense faster |

#### SmolLM2 360M on MPS

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Result |
|---|---:|---:|---|
| `256` | `41.79` | `494.46` | Dense faster |
| `512` | `39.99` | `375.00` | Dense faster |
| `1024` | `41.65` | `399.80` | Dense faster in the early frontier |
| `1536` | `172.39` | `456.58` | Dense faster in the early frontier |
| `2048` | `517.41` | `402.70` | DotCache faster on the trusted standalone rerun |

#### SmolLM2 360M on CUDA

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | Result |
|---|---:|---:|---|
| `2048` | `51.64` | `436.20` | Dense faster |
| `4096` | `38.95` | `655.83` | Dense faster |

#### Qwen2.5 3B on MPS

| Prompt Len | Dense Decode ms/step | DotCache Decode ms/step | KV Ratio | Read |
|---|---:|---:|---:|---|
| `256` | `1658.82` | `7386.23` | `0.68x` | Runs, but slow |
| `512` | `1073.71` | `7156.34` | `0.44x` | Runs, still very slow |
| `1024` | `74658.40` | `9637.94` | `1.71x` | Completes, but the dense baseline is noisy; do not treat this as a trusted crossover |

### Model-path optimization progression on MPS

The strongest exact MPS path improvements came from:

- batching decode by KV-head groups inside each layer
- keeping tails resident on device
- moving append and prefill ingest on-device where possible
- prewarming deferred prefill pages once
- reducing repeated grouped-decode validation
- adding selective static prepared-chunk reuse

Those changes are what moved SmolLM2 `2048` on MPS from an initial losing exact path to a real decode-speed win over dense on the best trusted rerun.

### Explicit dequantized baseline latency

The repo still uses the explicit dequantized path as a correctness oracle, but it is not yet maintained as a single first-class top-line latency metric in the model reports. The current performance comparison is therefore:

- dense KV
- DotCache exact compressed-domain path
- standalone microbench decode ladders

Adding a stable explicit-dequant end-to-end latency column would be a good follow-up to this report.

### Crossover context lengths

- TinyLlama MPS, KV memory: crossover between about `73` and `145` prompt tokens
- TinyLlama MPS, decode speed: no trusted crossover on the tested exact path
- TinyLlama CUDA, KV memory: crossover by `289` tokens
- TinyLlama CUDA, decode speed: no crossover on the tested exact path
- SmolLM2 MPS, KV memory: already favorable by `256` tokens
- SmolLM2 MPS, decode speed: current trusted crossover is around the exact `2048`-token point on this M4
- SmolLM2 CUDA, decode speed: no crossover on the tested `2048` and `4096` points

## Observations

### Where DotCache wins

- KV-cache memory wins are already strong and repeatable once prompts are more than a small number of pages.
- Exact MPS decode can beat dense at higher context on the current SmolLM2 `2048` benchmark.
- The page format and compressed-domain execution contract are stable enough to support both MPS and CUDA backends plus real-model decode integration.

### Where DotCache loses

- Short-prompt model inference still pays too much control/setup overhead.
- TinyLlama exact decode remains slower than dense on both MPS and CUDA in the recorded runs.
- CUDA is currently a correctness/parity backend, not a performance-leading one.
- Prefill ingest and page preparation are still expensive at longer contexts, especially before warmup and caching effects settle.
- Qwen2.5 3B can run on this Mac for small exact-length smokes, but it is not a good box for sustained 3B-class optimization work.

### Likely causes

- the prototype still has eager kernel structure and Python/runtime orchestration overhead
- grouped static-page reuse helps, but only after the runtime has enough stable prefill pages to amortize setup
- CUDA has not yet had the deeper optimization passes that the MPS path already received
- the model-path benchmark still includes framework overhead outside the raw compressed-domain kernels
- `M1` and `M2` quality is model-dependent in a way that raw max-logit drift overstates on TinyLlama and still understates on SmolLM2 unless we also track teacher-forced loss

### Next experiments

- add a stable explicit-dequant end-to-end comparison column to the report
- optimize large-context prefill preparation and scheduling further, especially on CUDA
- continue exact decode optimization in the midrange SmolLM2 `512-1536` region where memory wins exist but speed wins are mixed
- run the new Phase 6 vLLM offline benchmark on the CUDA cloud instance and add the first dense-vs-shadow-vs-active records
- use the teacher-forced loss harness as the default local quality gate for any future `M1/M2` codec changes before promoting them into the main model path

## Phase 6 Status

The repo now contains a first vLLM adapter and offline benchmark harness in [bench_vllm_offline.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_vllm_offline.py) backed by [dotcache/integrations/vllm_adapter](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/vllm_adapter).

What is implemented:

- block-table-aligned page mirroring with `tokens_per_page == block_size`
- `dense`, `dotcache_shadow`, and `dotcache_active` mode wiring
- finalized-block persistence plus one live partial block
- a version guard for the pinned `vLLM 0.18.x` line
- local unit coverage for block ownership, finalized/live parity, and shadow-vs-active consistency

What is still missing from this report:

- real CUDA/vLLM benchmark numbers from the cloud instance
- active-mode latency comparison against stock vLLM dense attention
- shadow-mode logit-drift numbers from real vLLM runs

## Bottom Line

The DotCache MVP claim is now partially validated in software:

- compressed KV can be executed directly without widening full pages by default
- correctness is strong enough for exact end-to-end Llama-family decode on real checkpoints
- KV-memory savings are already substantial on real workloads
- decode-speed wins are possible on Apple MPS at higher context, but they are not yet universal across models, prompt lengths, or backends

That is a good benchmark-report outcome for the stage we are in: the execution-format idea is real, the memory story is strong, and the remaining work is largely systems optimization and runtime integration rather than basic feasibility.
