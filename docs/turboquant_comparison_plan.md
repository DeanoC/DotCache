# TurboQuant Comparison Plan

This document defines the first fair DotCache vs TurboQuant comparison.

It is intentionally narrow:

- same machine class
- same model
- same context lengths
- same quality metrics
- same KV-memory reporting surface

The goal is not to "win the benchmark" by mixing unlike setups. The goal is to learn:

1. where TurboQuant is stronger
2. where DotCache is stronger
3. which ideas are worth porting back into DotCache

## Source Baseline

TurboQuant reference branch:

- repo: [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda/tree/feature/turboquant-kv-cache)
- benchmark summary: [benchmark-results.md](https://raw.githubusercontent.com/spiritbuun/llama-cpp-turboquant-cuda/feature/turboquant-kv-cache/benchmark-results.md)
- quality gate: [scripts/turbo-quality-gate.sh](https://raw.githubusercontent.com/spiritbuun/llama-cpp-turboquant-cuda/feature/turboquant-kv-cache/scripts/turbo-quality-gate.sh)

Important reference claims from that branch:

- `turbo3` quality depends strongly on FWHT rotation plus norm correction
- `turbo3` and `turbo4` are compared primarily against `q8_0`
- their own first-line quality gate is:
  - perplexity at `2K`
  - decode speed at `4K`, `16K`, `32K`
- their recommended CUDA config is currently layer-adaptive `turbo3`, not uniform `turbo4`
- they explicitly report that asymmetric K-only or V-only promotion is usually worse than promoting both inside a layer

## Fairness Rules

For a comparison to count, all of these should match:

- identical checkpoint
- identical prompt or eval text
- identical context length
- identical decode step count
- identical precision for the dense baseline
- identical measurement definition

We should report two separate comparisons instead of blending them:

1. **External baseline comparison**
   - DotCache vs TurboQuant vs dense/q8 baseline on the CUDA box
   - purpose: practical competitiveness

2. **Mechanism comparison**
   - DotCache `T3` / `M0` / `M1` / `M2` vs TurboQuant ideas
   - purpose: understand which codec ideas transfer cleanly

Do not compare:

- MPS results directly against their CUDA numbers
- TinyLlama against their Qwen3.5 27B results
- Hugging Face dense KV bytes against llama.cpp `q8_0` KV bytes without clearly labeling the backend/runtime difference

## First Comparison Matrix

### Target platform

- NVIDIA CUDA cloud instance
- one machine profile per report
- record GPU model, VRAM, driver, CUDA version, torch version, and vLLM version if involved

### Target models

Required:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Optional later:

- one larger `4K+` or `8K+` Llama-family model that fits both runtimes comfortably

### Target configurations

Dense / baseline:

- Hugging Face dense KV
- DotCache exact `M0`
- TurboQuant `q8_0` baseline

TurboQuant:

- `turbo3 uniform`
- `turbo3 LA-1`
- `turbo3 LA-5`
- `turbo4 uniform` optional

DotCache:

- exact `M0`
- experimental `T3`
- `V`-only `M1` optional
- fixed segmented `K=M2, V=M0` optional

The default first fair comparison should be:

- Dense baseline
- DotCache exact `M0`
- DotCache experimental `T3`
- TurboQuant `q8_0`
- TurboQuant `turbo3 uniform`
- TurboQuant `turbo3 LA-1`
- TurboQuant `turbo3 LA-5`

That gives one exact DotCache baseline, one Turbo-style DotCache reference, and the three most relevant TurboQuant operating points.

## Metrics To Collect

### Quality

Required:

- teacher-forced loss
- teacher-forced perplexity
- loss delta vs dense/q8 baseline
- token agreement
- max abs logit drift

Optional:

- long-context perplexity sweep at `2K`, `4K`, `8K`

### Performance

Required:

- prefill tokens/s or prefill ms for a fixed prompt length
- decode tokens/s or decode ms/step
- prompt length
- decode steps

### Memory

Required:

- KV-cache bytes if directly measurable
- peak GPU memory for the run
- whether the baseline OOMs

Optional:

- resident prepared-page bytes inside DotCache
- host-to-device bytes for DotCache prefill/decode

## Recommended Workloads

### Teacher-forced local-style check

Use the DotCache loss harness shape:

- TinyLlama: `sequence_length=320`, `prefix_length=288`, `eval_steps=32`
- SmolLM2: `sequence_length=1040`, `prefix_length=1024`, `eval_steps=16`

This gives direct continuity with the existing journal.

### Decode-speed ladder

Collect:

- `4K`
- `16K`
- `32K`

This mirrors TurboQuant's own gate.

### Long-context fit check

Collect:

- largest prompt that fits for each config
- first OOM point

That matters because TurboQuant emphasizes context-capacity wins as much as throughput.

## Result Table Template

Fill one row per `(model, runtime, config, context)`:

| Model | Runtime | Config | Context | Prefill | Decode | Loss Delta | PPL Ratio | Token Agr. | KV Bytes | Peak GPU Mem | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

Where:

- `Runtime` is one of `hf_dense`, `dotcache`, `llama.cpp_turboquant`
- `Config` is explicit, for example `M0 exact`, `T3`, `q8_0`, `turbo3 LA-1`
- `Status` is `ok`, `oom`, `quality_fail`, or `not_run`

## Acceptance Gates

For the first report, success means:

- we can produce a table filled from real runs
- the same model has been measured in both systems
- we can state one honest takeaway on:
  - quality
  - decode speed
  - context capacity

It does **not** require DotCache to beat TurboQuant.

## Expected Early Read

Based on current evidence, the likely first outcome is:

- TurboQuant will be stronger on CUDA throughput and extreme-context fit
- DotCache exact `M0` will be the cleaner correctness baseline
- DotCache `T3` will likely lose badly at first and serve mainly as a mechanism comparison
- the interesting question is whether TurboQuant's winning ingredients are:
  - FWHT rotation
  - norm correction
  - layer-adaptive promotion
  - CUDA kernel strategy

That distinction matters because only some of those ideas are codec ideas we can transplant into DotCache.

## Immediate Next Runs

1. Reproduce TurboQuant's own quality gate on the CUDA box.
2. Run the matching DotCache CUDA loss/decode harnesses on the same model and contexts.
3. Fill the result table for `TinyLlama` first.
4. Repeat on `SmolLM2`.
5. Only after that, judge whether a deeper Turbo3/Turbo4 port is worth pursuing locally.
