# Qwen3.5 0.8B Full Sweep Summary

This document is the checked-in summary for the full `Qwen/Qwen3.5-0.8B` CUDA comparison sweep run on `2026-03-29`.

It is the first single-directory snapshot in this repo that includes all of the following on the same prompt ladder:

- shared native dense baseline
- native `StateCache`
- native hybrid `DotCache + StateCache`
- external `llama.cpp` TurboQuant

The raw artifacts live in [benchmarks/results/qwen35_context_sweep_20260329_full](/workspace/DotCache/benchmarks/results/qwen35_context_sweep_20260329_full).

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

Memory matrix:

```bash
python scripts/report_turboquant_comparison.py \
  --layout memory_matrix \
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
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `q8_0` | `tok/s` | `242.10` | `329.40` | `325.80` | `323.40` | `286.90` | `310.30` | `288.10` | `239.60` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_uniform` | `tok/s` | `283.70` | `292.50` | `285.30` | `288.20` | `268.90` | `249.80` | `228.40` | `185.70` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_la1` | `tok/s` | `283.40` | `299.00` | `298.30` | `297.10` | `286.80` | `268.80` | `245.40` | `202.10` |

## Memory

Units are `MiB`.

| Model | Runtime | Config | Unit | 448 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | `hf_dense` | `dense (shared harness)` | `MiB` | `24.13` | `30.88` | `42.88` | `66.88` | `114.88` | `210.88` | `402.88` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `statecache_hf` | `StateCache M0 8-bit` | `MiB` | `11.77` | `18.52` | `30.52` | `54.52` | `102.52` | `198.52` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `hybrid_dotcache_statecache_hf` | `Hybrid DotCache+StateCache` | `MiB` | `11.28` | `17.19` | `27.69` | `48.69` | `89.63` | `171.38` | `OOM` | `OOM` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `q8_0` | `MiB` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_uniform` | `MiB` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| `Qwen/Qwen3.5-0.8B` | `llama.cpp_turboquant` | `turbo3_la1` | `MiB` | `-` | `-` | `-` | `-` | `-` | `-` | `-` | `-` |

## Readout

- `StateCache` is still the strongest native path through `16384`. It stays close to `60 tok/s` while using less resident memory than the shared dense baseline at every measured context.
- The shared dense baseline changes the ceiling story. Plain dense survives `32768` on this pod at `58.97 tok/s` and `402.88 MiB`, while the StateCache and hybrid harnesses both hit CUDA `OOM` at `32768`.
- The combined hybrid lane remains memory-cheaper than dense, but its decode rate collapses as context grows. By `16384` it is down to `0.81 tok/s`, so it is not a competitive native serving point yet.
- TurboQuant remains far faster than every native path and continues to run at `32768` and `65536`, where the native StateCache and hybrid rows already fail.
- TurboQuant still lacks a comparable checked-in total-memory metric in this repo, so the external rows are throughput-only in the memory table.

## Important Fairness Notes

- The native dense row is now the shared dense harness, not one of the capture-heavy benchmark harnesses. That removes the earlier misleading comparison between `dense (hybrid harness)` and `dense (statecache harness)`.
- TurboQuant prompt construction is now exact-token for the requested lengths and no longer passes long prompts through argv. The `32768` and `65536` external rows are real model runs, not shell-limit artifacts.
- This is still not a perfect mechanism-equivalence comparison. `StateCache` is compressing native Qwen3.5 recurrent state inside the Hugging Face runtime, while TurboQuant is an external GGUF / `llama.cpp` KV-quantized serving stack.

## Bottom Line

- For native Hugging Face Qwen3.5 serving on this pod, `StateCache M0 8-bit` is the keepable result.
- For maximum throughput and long-context survival, the external TurboQuant lane is decisively ahead.
- The combined native hybrid lane is now a measured data point, but it is not close to production-worthy yet.
