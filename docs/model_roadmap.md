# Model Roadmap

This document is the scaffold for "proper model" work beyond the current TinyLlama and SmolLM2 local lanes.

## Lanes

- `HF native + DotCache`
  - Main engineering lane for exact DotCache integration.
  - Best next targets: Llama 3.2 3B, then Qwen2.5 3B.
- `GGUF / llama.cpp`
  - External reference lane for memory, latency, and TurboQuant-style comparisons.
  - Useful for comparison, not the primary integration surface.
- `Qwen3.5 hybrid`
  - Reference-only for now.
  - Not a next-step target because it is not a plain Llama-style decoder path.

## Recommended Order

1. `meta-llama/Llama-3.2-3B-Instruct`
2. `Qwen/Qwen2.5-3B-Instruct`
3. `Qwen/Qwen2.5-7B-Instruct` on CUDA
4. `Qwen/Qwen3.5-*` only after a non-Llama DotCache abstraction exists

## Shared Matrix

The canonical scaffold for model selection and benchmark planning is:

- registry: [dotcache/model_registry.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/model_registry.py)
- matrix CLI: [benchmarks/bench_model_matrix.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_model_matrix.py)

Use the matrix CLI to:

- list all planned model targets
- emit runnable compare commands for models already supported by the HF DotCache path
- keep HF native and GGUF reference lanes in one place even before every runtime path exists

Example:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --output-format pretty
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys tinyllama_hf smollm2_360m_hf --run-supported --backend torch_mps --device mps
```

## Current Local Read

- This Mac already has working local lanes for:
  - TinyLlama HF
  - SmolLM2 360M HF
- The next local "proper model" step should be:
  - Llama 3.2 3B HF first
  - Qwen2.5 3B HF second
- GGUF should be treated as an external baseline lane, not as a replacement for the native-weight DotCache path.

## First Proper-Model Lane

Llama 3.2 3B is now the first-class stretch-model target on the existing HF Llama path:

- wrapper: [scripts/run_llama32_compare.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_llama32_compare.sh)
- registry entry: [dotcache/model_registry.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/model_registry.py)
- harness: [benchmarks/bench_llama_compare.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_compare.py)

Use it directly on this Mac with:

```bash
bash scripts/run_llama32_compare.sh
```

Or through the shared matrix:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf --output-format pretty
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf --run-supported --backend torch_mps --device mps
```

The matrix now passes `--continue-on-error` through to runnable compare harnesses by default so stretch-model lanes can be exercised without treating a single OOM or gated-model failure as a framework bug.
