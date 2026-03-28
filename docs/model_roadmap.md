# Model Roadmap

This document is the scaffold for "proper model" work beyond the current TinyLlama and SmolLM2 local lanes, with the RTX 5090 CUDA pod as the first larger-model scale-up box.

## Lanes

- `HF native + DotCache`
  - Main engineering lane for exact DotCache integration.
  - Best guaranteed next targets: Qwen2.5 3B, then Qwen2.5 7B on CUDA.
  - Llama 3.2 3B stays a first-class optional lane when access exists.
- `GGUF / llama.cpp`
  - External reference lane for memory, latency, and TurboQuant-style comparisons.
  - Useful for comparison, not the primary integration surface.
- `Qwen3.5 hybrid`
  - Reference-only for now.
  - Not a next-step target because it is not a plain Llama-style decoder path.

## Recommended Order

1. `Qwen/Qwen2.5-3B-Instruct` on CUDA
2. `Qwen/Qwen2.5-7B-Instruct` on CUDA
3. `meta-llama/Llama-3.2-3B-Instruct` on CUDA when access exists
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
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys qwen25_3b_hf qwen25_7b_hf --run-supported --backend torch_cuda --device cuda
```

The matrix now also emits runnable external GGUF reference commands for `llama.cpp` lanes:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_gguf qwen25_3b_gguf qwen25_7b_gguf --output-format pretty
```

And it can emit optional mounted-HF commands for large native-weight repos:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf qwen25_3b_hf --mount-hf-models --output-format pretty
```

Recorded runs can then be summarized with:

```bash
.venv/bin/python scripts/report_model_benchmarks.py --benchmark qwen2_compare
.venv/bin/python scripts/report_model_benchmarks.py --benchmark llama_compare
```

Canonical 5090-era Qwen CUDA labels:

```bash
.venv/bin/python scripts/record_benchmark.py --label qwen25-3b-cuda -- bash scripts/run_qwen25_compare_cuda.sh --default-mode-k M0 --default-mode-v M0
.venv/bin/python scripts/record_benchmark.py --label qwen25-3b-cuda-k-exact -- bash scripts/run_qwen25_compare_cuda.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-3b-cuda-selective -- bash scripts/run_qwen25_compare_cuda_selective.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda -- bash scripts/run_qwen25_7b_compare_cuda.sh --default-mode-k M0 --default-mode-v M0
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda-k-exact -- bash scripts/run_qwen25_7b_compare_cuda.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda-selective -- bash scripts/run_qwen25_7b_compare_cuda_selective.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda-planner-aggressive -- bash scripts/run_qwen25_7b_compare_cuda_planner_aggressive.sh
```

Use the unlabeled wrapper defaults for the recommended path. Add the explicit `--default-mode-k M0 --default-mode-v M0` override only when you want the Qwen CUDA baseline lane for comparison.

## Current Read

- This Mac already has working local lanes for:
  - TinyLlama HF
  - SmolLM2 360M HF
- The RTX 5090 pod is now the preferred larger-model lane for:
  - Qwen2.5 3B HF first
  - Qwen2.5 7B HF second
  - Llama 3.2 3B HF when access exists
- GGUF should be treated as an external baseline lane, not as a replacement for the native-weight DotCache path.

## CUDA Pod Lane

The first pod-oriented HF scale-up lane should use the existing compare harnesses on `torch_cuda`:

- public 3B wrapper: [scripts/run_qwen25_compare_cuda.sh](/workspace/DotCache/scripts/run_qwen25_compare_cuda.sh)
- selective 3B wrapper: [scripts/run_qwen25_compare_cuda_selective.sh](/workspace/DotCache/scripts/run_qwen25_compare_cuda_selective.sh)
- public 7B wrapper: [scripts/run_qwen25_7b_compare_cuda.sh](/workspace/DotCache/scripts/run_qwen25_7b_compare_cuda.sh)
- selective 7B wrapper: [scripts/run_qwen25_7b_compare_cuda_selective.sh](/workspace/DotCache/scripts/run_qwen25_7b_compare_cuda_selective.sh)
- planner-aggressive 7B wrapper: [scripts/run_qwen25_7b_compare_cuda_planner_aggressive.sh](/workspace/DotCache/scripts/run_qwen25_7b_compare_cuda_planner_aggressive.sh)
- Qwen key-exact research wrappers:
  - [scripts/run_qwen25_compare_cuda_k_exact.sh](/workspace/DotCache/scripts/run_qwen25_compare_cuda_k_exact.sh)
  - [scripts/run_qwen25_7b_compare_cuda_k_exact.sh](/workspace/DotCache/scripts/run_qwen25_7b_compare_cuda_k_exact.sh)
- optional gated Llama wrapper: [scripts/run_llama32_compare_cuda.sh](/workspace/DotCache/scripts/run_llama32_compare_cuda.sh)

Use the public path first:

```bash
bash scripts/run_qwen25_compare_cuda.sh
bash scripts/run_qwen25_7b_compare_cuda.sh
```

Those public Qwen CUDA wrappers now default to `K=M3 / V=M0`.

Then add the optional Llama lane if available:

```bash
bash scripts/run_llama32_compare_cuda.sh
```

## First Proper-Model Lane

Llama 3.2 3B remains the first-class gated stretch-model target on the existing HF Llama path:

- wrapper: [scripts/run_llama32_compare.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_llama32_compare.sh)
- registry entry: [dotcache/model_registry.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/model_registry.py)
- harness: [benchmarks/bench_llama_compare.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_compare.py)

Use it directly on the local MPS lane with:

```bash
bash scripts/run_llama32_compare.sh
```

Or through the shared matrix:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf --output-format pretty
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf --run-supported --backend torch_mps --device mps
```

The matrix now passes `--continue-on-error` through to runnable compare harnesses by default so stretch-model lanes can be exercised without treating a single OOM or gated-model failure as a framework bug.

## Optional hf-mount Lane

For large Hub repos where disk/download friction is the problem, there is now an optional `hf-mount` scaffold:

- runner: [benchmarks/bench_hf_mount_compare.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_hf_mount_compare.py)
- wrappers:
  - [scripts/run_llama32_compare_mounted.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_llama32_compare_mounted.sh)
  - [scripts/run_qwen25_compare_mounted.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen25_compare_mounted.sh)

This lane is intentionally about fetch ergonomics, not runtime magic:

- it mounts a Hub repo as a local filesystem with `hf-mount`
- it runs the normal HF compare harness against the mounted path
- it helps when full repo download size is the main annoyance
- it does not change the real RAM / VRAM / decode-time limits of the model itself

## First Non-Llama Native-Weight Lane

Qwen2.5 3B is now the first non-Llama native-weight DotCache target on the HF path:

- wrapper: [scripts/run_qwen25_compare.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen25_compare.sh)
- harness: [benchmarks/bench_qwen2_compare.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen2_compare.py)
- adapter: [dotcache/integrations/qwen2.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/qwen2.py)

This is intentionally a narrow architecture expansion:

- Qwen2-family only
- reuses the existing replay/generation/loss harness functions
- adds a Qwen2-specific attention wrapper instead of pretending the Llama wrapper is universal

Run it directly with:

```bash
bash scripts/run_qwen25_compare.sh
```

For the CUDA pod version of the same lane, use:

```bash
bash scripts/run_qwen25_compare_cuda.sh
```

## First Larger Public CUDA Lane

Qwen2.5 7B is now the first larger public model lane for the 5090 pod:

- wrapper: [scripts/run_qwen25_7b_compare_cuda.sh](/workspace/DotCache/scripts/run_qwen25_7b_compare_cuda.sh)
- harness: [benchmarks/bench_qwen2_compare.py](/workspace/DotCache/benchmarks/bench_qwen2_compare.py)
- registry entry: [dotcache/model_registry.py](/workspace/DotCache/dotcache/model_registry.py)

This lane intentionally reuses the existing Qwen2 adapter rather than expanding the architecture surface:

- same `qwen2_compare` harness
- CUDA default is now `K=M3 / V=M0`
- same `1024 2048 4096` prompt grid
- same `--continue-on-error` stretch-model behavior

Current 5090-era research note:

- Qwen2.5 3B on CUDA is materially more stable with `K=M3 / V=M0` than with default `M0/M0`.
- Qwen2.5 7B on CUDA shows the same pattern at `1024/2048`.
- The repo now treats `K=M3 / V=M0` as the recommended Qwen CUDA default, while keeping `M0/M0` as the baseline comparison lane.
- The best current low-memory adaptive 7B lane is now the planner-driven aggressive policy, not the older fixed selective override wrapper.

## Selective Key Precision

DotCache now has the beginnings of a more general capability than “pick one global key mode for the whole model”:

- Selective Key Precision
- a model-specific Sensitivity Map can keep exact key pages only in fragile layers or KV groups
- the rest of the key cache can stay on `M0`
- value pages can remain compressed independently

The first validated case is Qwen2.5 3B on CUDA. Offline score probes and live benchmark runs both point to:

- broad early fragility at `layer 0`
- a narrower late hotspot at `layer 27`, mostly `KV1`

That gives a first practical selective policy:

- `layer:0=M3`
- `layer:27:kv:1=M3`
- all other K pages stay `M0`
- V stays `M0`

The current selective wrapper is:

```bash
bash scripts/run_qwen25_compare_cuda_selective.sh
```

This should be treated as a reusable capability, not just a Qwen quirk. The next roadmap step is to build sensitivity maps for other models and check whether:

- some models want whole-layer exact K
- some want only specific KV groups
- some still need full `K=M3`

The same lightweight policy already transfers to Qwen2.5 7B at `1024/2048`:

- `layer:0=M3`
- `layer:27:kv:1=M3`
- greedy agreement returns to `1.0`

That is no longer the most useful low-memory 7B lane on this pod. The current better adaptive wrapper is the true per-page planner path:

```bash
bash scripts/run_qwen25_7b_compare_cuda_planner_aggressive.sh
```

At exact `4096` on CUDA this planner-aggressive lane currently:

- keeps greedy agreement at `1.0`
- reduces KV resident bytes below the fixed selective wrapper
- stays slower than exact-K, so it should be treated as the memory-first 7B lane rather than the default runtime lane
- KV ratio stays near the all-`M0` lane rather than the full exact-K lane

## GGUF Reference Lane

The GGUF / `llama.cpp` side now has a minimal external benchmark scaffold:

- runner: [benchmarks/bench_gguf_external.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_gguf_external.py)
- wrappers:
  - [scripts/run_llama32_gguf_reference.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_llama32_gguf_reference.sh)
  - [scripts/run_qwen25_gguf_reference.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen25_gguf_reference.sh)
  - [scripts/run_qwen25_7b_gguf_reference.sh](/workspace/DotCache/scripts/run_qwen25_7b_gguf_reference.sh)

It is intentionally an external reference lane, not a DotCache integration:

- it builds exact-length prompts with the matching Hugging Face tokenizer
- it calls `llama-cli -hf ...`
- it parses `llama.cpp` timing lines when present
- it degrades cleanly with an error record when `llama-cli` is unavailable

This gives the shared matrix one consistent way to emit or run:

- HF dense / DotCache lanes
- GGUF / `llama.cpp` reference lanes
- future TurboQuant-style external baselines
