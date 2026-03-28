# CUDA Next Steps

This note is the shortest useful handoff from the local Apple MPS exploration to the active CUDA box.

The key local hints worth carrying forward are:

- `M0 3b` is now a real intermediate tier, and `K=4b, V=3b` is the most plausible first CUDA probe.
- `M3 int8` now works end to end and the hierarchical planner can emit recent sealed pages as `M3:int8`.
- TinyLlama has one clearly useful local adaptive profile.
- SmolLM2 does not yet have a clearly good adaptive profile, but it does have a safer starting point.

## Starter Profiles

Use these first:

- [tinyllama_cuda_start.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_cuda_start.yaml)
- [smollm2_360m_cuda_start.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_cuda_start.yaml)

## Recommended Run Order

### TinyLlama

1. Exact baseline:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --target-prompt-lengths 577 \
  --max-new-tokens 4
```

2. Local-profile adaptive policy:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layer-profile configs/layer_profiles/tinyllama_cuda_start.yaml \
  --target-prompt-lengths 577 \
  --max-new-tokens 4
```

3. Exact `K=4b, V=3b` probe:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --bits-k 4 \
  --bits-v 3 \
  --target-prompt-lengths 577 \
  --max-new-tokens 4
```

4. Adaptive policy plus recent-page `M3 int8`:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layer-profile configs/layer_profiles/tinyllama_cuda_start.yaml \
  --recent-page-escape-dtype int8 \
  --target-prompt-lengths 577 \
  --max-new-tokens 4
```

### SmolLM2 360M

1. Exact baseline:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error
```

2. Safer local adaptive profile:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --layer-profile configs/layer_profiles/smollm2_360m_cuda_start.yaml \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error
```

3. Exact `K=4b, V=3b` probe:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --bits-k 4 \
  --bits-v 3 \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error
```

4. Safer adaptive profile plus recent-page `M3 int8`:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py \
  --backend torch_cuda \
  --device cuda \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --layer-profile configs/layer_profiles/smollm2_360m_cuda_start.yaml \
  --recent-page-escape-dtype int8 \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error
```

## What To Watch

- teacher-forced loss before greedy agreement
- per-mode page counts and fragmentation metrics
- whether `K=4b, V=3b` is quality-safe enough to promote into adaptive candidate sets
- whether recent-page `M3 int8` keeps its live-tail memory win without paying as much runtime cost as it did on MPS

## Current Local Read

- TinyLlama:
  - best adaptive profile is still the first-pass local profile
  - aggressive values look useful
  - extra early-key aggression was not worthwhile
- SmolLM2:
  - the safer second/third-pass family is the right CUDA starting point
  - late-value overrides should stay off until better evidence exists
  - the current `M2` family still looks like the limiting factor
