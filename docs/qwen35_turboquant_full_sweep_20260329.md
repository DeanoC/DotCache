# Qwen3.5 0.8B Full Sweep Summary

This document is the checked-in summary for the full `Qwen/Qwen3.5-0.8B` CUDA comparison sweep run on `2026-03-29`.

It is the first single-directory snapshot in this repo that includes all of the following on the same prompt ladder:

- shared native dense baseline
- native `StateCache`
- native hybrid `DotCache + StateCache`
- external `llama.cpp` TurboQuant

The raw artifacts live in [benchmarks/results/qwen35_context_sweep_20260329_full](/workspace/DotCache/benchmarks/results/qwen35_context_sweep_20260329_full).

Follow-on artifacts for the quality and serving-only checks live in:

- [qwen35_quality_sweep_20260329](/workspace/DotCache/benchmarks/results/qwen35_quality_sweep_20260329)
- [qwen35_statecache_serving_20260329](/workspace/DotCache/benchmarks/results/qwen35_statecache_serving_20260329)

## Scope

- model: `Qwen/Qwen3.5-0.8B`
- native runtime: Hugging Face / PyTorch CUDA
- external runtime: `llama.cpp` TurboQuant CUDA fork on GGUF weights
- prompt lengths: `448`, `1024`, `2048`, `4096`, `8192`, `16384`, `32768`, `65536`
- decode length: `4` generated tokens

The native dense row now comes from the shared plain-text harness instead of the heavier capture harnesses, so the dense baseline is much closer to apples-to-apples with the rest of the native rows.

## Commands

Full sweep:

```bash
PYTHONPATH=/workspace/DotCache .venv/bin/python scripts/run_qwen35_context_sweep.py \
  --contexts 448 1024 2048 4096 8192 16384 32768 65536 \
  --output-dir benchmarks/results/qwen35_context_sweep_20260329_full \
  --dense-timeout-seconds 1800 \
  --statecache-timeout-seconds 1800 \
  --hybrid-timeout-seconds 1800 \
  --turboquant-timeout-seconds 1800
```

Throughput matrix:

```bash
python scripts/report_turboquant_comparison.py \
  --layout context_matrix \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_dense_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_statecache_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_hybrid_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_turboquant_sweep.jsonl
```

Cache/state memory matrix:

```bash
python scripts/report_turboquant_comparison.py \
  --layout cache_memory_matrix \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_dense_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_statecache_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_hybrid_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_turboquant_sweep.jsonl
```

Total device memory matrix:

```bash
python scripts/report_turboquant_comparison.py \
  --layout device_memory_matrix \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_dense_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_statecache_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_hybrid_sweep.jsonl \
  --input benchmarks/results/qwen35_context_sweep_20260329_full/qwen35_0p8b_turboquant_sweep.jsonl
```

## Throughput

Units are `tok/s`.

| Model | Runtime | Config | Unit | 448 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | `hf_dense` | `dense (shared harness)` | `tok/s` | `18.75` | `62.66` | `61.84` | `61.54` | `61.19` | `61.11` | `58.97` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `statecache_hf` | `StateCache M0 8-bit` | `tok/s` | `59.95` | `59.80` | `54.08` | `60.58` | `60.20` | `60.78` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `hybrid_dotcache_statecache_hf` | `Hybrid DotCache+StateCache` | `tok/s` | `8.87` | `7.66` | `5.04` | `2.99` | `1.62` | `0.81` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `q8_0` | `tok/s` | `195.80` | `326.20` | `316.40` | `320.70` | `326.40` | `288.70` | `280.60` | `244.90` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_uniform` | `tok/s` | `281.50` | `282.10` | `275.80` | `262.80` | `281.70` | `250.60` | `227.50` | `187.60` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_la1` | `tok/s` | `297.80` | `301.50` | `291.10` | `297.20` | `275.60` | `258.20` | `235.60` | `208.80` |

## Memory

There are now two memory tables because one mixed table was misleading.

- `Cache/state memory` is the closest thing to a mechanism comparison:
  - native dense: final native cache bytes
  - native StateCache: compressed native state bytes
  - native hybrid: resident DotCache pages plus fixed StateCache bytes
  - TurboQuant: `llama.cpp` context bytes
- `Total device memory` is a deployment-envelope view:
  - native rows use peak CUDA allocated bytes when the harness recorded them
  - TurboQuant rows use `llama.cpp` device `self` bytes

Those are not the same thing, so they should not be read as one unified “memory winner” table.

### Cache/State Memory

Units are `MiB`.

| Model | Runtime | Config | Unit | 448 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | `hf_dense` | `dense (shared harness)` | `MiB` | `24.13` | `30.88` | `42.88` | `66.88` | `114.88` | `210.88` | `402.88` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `statecache_hf` | `StateCache M0 8-bit` | `MiB` | `11.77` | `18.52` | `30.52` | `54.52` | `102.52` | `198.52` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `hybrid_dotcache_statecache_hf` | `Hybrid DotCache+StateCache` | `MiB` | `11.28` | `17.19` | `27.69` | `48.69` | `89.63` | `171.38` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `q8_0` | `MiB` | `1651.00` | `1651.00` | `1651.00` | `1651.00` | `1651.00` | `1651.00` | `1651.00` | `1651.00` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_uniform` | `MiB` | `691.00` | `691.00` | `691.00` | `691.00` | `691.00` | `691.00` | `691.00` | `691.00` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_la1` | `MiB` | `1011.00` | `1011.00` | `1011.00` | `1011.00` | `1011.00` | `1011.00` | `1011.00` | `1011.00` |

This is the most honest “how much serving state are we carrying?” view in the repo today, but it is still cross-runtime:

- native rows are measured from native Qwen3.5 cache/state accounting
- TurboQuant rows come from `llama.cpp` context bytes, not native Hugging Face tensors

### Total Device Memory

Units are `MiB`.

| Model | Runtime | Config | Unit | 448 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | `hf_dense` | `dense (shared harness)` | `MiB` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `statecache_hf` | `StateCache M0 8-bit` | `MiB` | `2165.83` | `2756.75` | `3739.60` | `5751.67` | `9777.52` | `17833.71` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `hybrid_dotcache_statecache_hf` | `Hybrid DotCache+StateCache` | `MiB` | `-` | `-` | `-` | `-` | `-` | `-` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `q8_0` | `MiB` | `2932.00` | `2932.00` | `2932.00` | `2932.00` | `2932.00` | `2932.00` | `2932.00` | `2932.00` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_uniform` | `MiB` | `1972.00` | `1972.00` | `1972.00` | `1972.00` | `1972.00` | `1972.00` | `1972.00` | `1972.00` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_la1` | `MiB` | `2292.00` | `2292.00` | `2292.00` | `2292.00` | `2292.00` | `2292.00` | `2292.00` | `2292.00` |

This table is useful for deployment-envelope questions, but it is explicitly not a codec-only comparison:

- native dense is blank because that harness did not record peak CUDA memory in the checked-in run
- native StateCache is peak CUDA allocated memory from the native benchmark
- TurboQuant is `llama.cpp` device `self` bytes from its own memory breakdown

## Quality Gate

The native and external quality checks are now checked in separately because they are not the same metric surface:

- native: teacher-forced perplexity over `16` eval steps after a fixed prefix
- external: `llama-perplexity` over a `2 x context` text file with the same tokenizer family and context setting

### Native Teacher-Forced Perplexity

| Prefix | Dense PPL | StateCache PPL | Ratio |
|---|---:|---:|---:|
| `16384` | `1.0001547` | `1.0001527` | `0.9999980` |
| `32768` | `1.0001489` | `1.0001515` | `1.0000026` |

### External `llama-perplexity`

| Context | `q8_0` | `turbo3_uniform` | `turbo3_la1` |
|---|---:|---:|---:|
| `16400` | `1.0002` | `1.0007` | `1.0002` |
| `32784` | `1.0002` | `1.0007` | `1.0002` |

The practical read is that the native `StateCache` lane stays numerically indistinguishable from dense on these long-context checks, while the external TurboQuant lane keeps `q8_0`-level perplexity for `turbo3_la1` and shows a small but repeatable degradation for `turbo3_uniform`.

## `32768` OOM Root Cause

The full sweep originally suggested that native `StateCache` simply could not survive `32768`, but the serving-only check shows that is not true.

- compare/readout harness:
  - `StateCache` and hybrid both `OOM` at `32768`
- serving-only harness:
  - `StateCache` prefill succeeds at `32768`
  - `StateCache` first `OOM` is `65536`

That serving-only check is in [qwen35_0p8b_statecache_serving_xlarge.jsonl](/workspace/DotCache/benchmarks/results/qwen35_statecache_serving_20260329/qwen35_0p8b_statecache_serving_xlarge.jsonl). The `32768` row records `runtime_mode = statecache_serving_only`, `prefill_ms = 501.29`, and `hybrid_state_total_bytes = 422461440` (`402.88 MiB`), while the `65536` row is the first CUDA `OOM`.

So the corrected interpretation is:

- plain dense and serving-only `StateCache` both survive `32768` on this pod
- the earlier `32768` `StateCache` failure belongs to the heavier compare/readout harness, which duplicates dense capture and prefill state
- `65536` is the first native `StateCache` serving failure point we have measured here

## Readout

- Throughput: the TurboQuant speed lead is real for this external stack. The table is using decode-only throughput on both sides, so the gap is not just a reporting bug. But it is still a cross-runtime result, not proof that the TurboQuant codec alone is several times better than StateCache.
- `StateCache` is still the strongest native path through `16384`. It stays close to `60 tok/s` while using less resident memory than the shared dense baseline at every measured context.
- The shared dense baseline changes the ceiling story. Plain dense survives `32768` on this pod at `58.97 tok/s` and `402.88 MiB`. The compare/readout StateCache harness `OOM`s there, but the serving-only StateCache path does not fail until `65536`.
- The combined hybrid lane remains memory-cheaper than dense, but its decode rate collapses as context grows. By `16384` it is down to `0.81 tok/s`, so it is not a competitive native serving point yet.
- TurboQuant remains far faster than every native path and continues to run at `32768` and `65536`, where the native StateCache and hybrid rows already fail.
- Memory: the old single memory table was not fair. The corrected read is:
  - cache/state table: closest thing to a mechanism comparison, though still cross-runtime
  - total device table: deployment-envelope only
- TurboQuant’s context bytes are much larger than the native StateCache bytes, but its total device memory on this box is still flat and modest relative to native StateCache peak allocation: `q8_0 = 2932 MiB`, `turbo3_la1 = 2292 MiB`, `turbo3_uniform = 1972 MiB`.
- The long-context quality gate is now filled in. Native `StateCache` stays essentially identical to dense, while external `turbo3_la1` matches `q8_0` perplexity and `turbo3_uniform` is slightly worse.

## Important Fairness Notes

- The native dense row is now the shared dense harness, not one of the capture-heavy benchmark harnesses. That removes the earlier misleading comparison between `dense (hybrid harness)` and `dense (statecache harness)`.
- TurboQuant prompt construction is now exact-token for the requested lengths and no longer passes long prompts through argv. The `32768` and `65536` external rows are real model runs, not shell-limit artifacts.
- The external quality gate uses `llama-perplexity`, which requires at least `2 x context` tokens for a one-chunk run. Those prompt files are generated separately from the native teacher-forced loss inputs.
- The throughput table is a real deployment comparison, but not a pure codec comparison. Native runs are Hugging Face / PyTorch CUDA; TurboQuant runs are GGUF + `llama.cpp` CUDA.
- The cache/state memory table and the total-device-memory table answer different questions and should not be collapsed into one score.
- This is still not a perfect mechanism-equivalence comparison. `StateCache` is compressing native Qwen3.5 recurrent state inside the Hugging Face runtime, while TurboQuant is an external GGUF / `llama.cpp` KV-quantized serving stack.

## Bottom Line

- If the question is “what is the strongest native Hugging Face result here?”, the answer is still `StateCache M0 8-bit`.
- If the question is “what stack is fastest and survives longest context on this box?”, the answer is the external TurboQuant `llama.cpp` lane.
- If the question is “have we proven TurboQuant’s codec is intrinsically better than StateCache in a pure apples-to-apples comparison?”, the answer is no.
- The combined native hybrid lane is now a measured data point, but it is not close to production-worthy yet.
