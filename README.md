# DotCache

DotCache is a compressed-domain KV-cache runtime for transformer decode research.

The repo is now beyond the original bootstrap phase. It currently includes:

- exact `M0` grouped affine quantization with `M3` escape/tail handling
- experimental approximate modes: `M1`, `M2`, and `Turbo3`
- compressed-domain `score` and `mix` execution
- `cpu_ref`, `torch_mps`, and `torch_cuda` backends
- Hugging Face integration for Llama-family and Qwen2-family checkpoints
- an early correctness-first `vLLM 0.18.x` adapter surface
- benchmark harnesses, model-matrix tooling, and a broad regression suite

## Current Read

DotCache is best understood today as a working compressed-domain KV-cache runtime and benchmark platform.

What is true right now:

- exact model-path decode works on real checkpoints
- KV-memory savings are already strong at meaningful context lengths
- MPS now has credible higher-context decode wins in selected exact runs
- CUDA already has real benchmark coverage and the intended memory behavior
- `vLLM` is in the adapter/offline-benchmark stage, not the production-runtime stage

What is not true yet:

- dense attention still wins many latency comparisons
- CUDA is not yet a speed-leading backend
- approximate modes are not ready to replace exact `M0` on the main model path

## Status Snapshot

- main supported path: exact-first `M0`
- experimental lanes: `M1`, `M2`, `Turbo3`
- model integrations: Llama-family and Qwen2-family
- accelerator backends: Apple Silicon `mps` and NVIDIA `cuda`
- local test status: `120 passed, 5 skipped`

## Benchmark Snapshot

### Memory

- TinyLlama `865` on MPS: DotCache uses about `0.29x` of dense KV bytes
- TinyLlama `577` on CUDA: DotCache uses about `0.36x` of dense KV bytes
- SmolLM2 `2048` on MPS: DotCache uses about `0.22x` of dense KV bytes
- SmolLM2 `4096` on CUDA: DotCache uses about `0.19x` of dense KV bytes

### Decode

- TinyLlama: dense still wins on the current exact path
- SmolLM2 on this M4: the strongest trusted exact point is `2048`, where DotCache beat dense on decode while keeping the KV footprint at about `0.22x`
- CUDA: exact decode works end to end with exact greedy agreement on the recorded runs, but it is still behind dense on latency

### CUDA Snapshot

The performance journal already includes real CUDA results on an NVIDIA RTX 2000 Ada machine:

- TinyLlama `289`: dense `22.31 ms/step`, DotCache `132.15 ms/step`, KV ratio `0.58x`
- TinyLlama `577`: dense `23.49 ms/step`, DotCache `134.48 ms/step`, KV ratio `0.36x`
- SmolLM2 `2048`: dense `51.64 ms/step`, DotCache `436.20 ms/step`, KV ratio `0.22x`
- SmolLM2 `4096`: dense `38.95 ms/step`, DotCache `655.83 ms/step`, KV ratio `0.19x`

So the right CUDA framing is: implemented, benchmarked, numerically stable, memory-efficient, and still waiting on deeper decode-kernel optimization.

### Approximate Modes

- TinyLlama is fairly forgiving for `V`-only `M1` and adaptive `K`-only `M2`
- SmolLM2 is not: teacher-forced loss/perplexity regress even when greedy agreement stays at `1.0`
- `Turbo3` is still a useful comparison lane, not a recommended default

For the compact benchmark summary, see [`docs/benchmark_report.md`](./docs/benchmark_report.md).
For the running log and latest checkpoints, see [`docs/performance_journal.md`](./docs/performance_journal.md).

## Quick Start

### Apple Silicon

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev,mps,hf]"
./scripts/run_unit_tests.sh
```

Tiny smoke benchmark:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend torch_mps --device mps --max-new-tokens 3
```

Best current higher-context local lane:

```bash
bash scripts/run_smollm2_frontier_compare.sh
```

### NVIDIA Linux

```bash
bash scripts/bootstrap_nvidia_llama_dev.sh
```

CUDA smoke benchmark:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend cpu_ref --device cuda --max-new-tokens 4
```

Real checkpoint:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --backend cpu_ref --device cuda --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Write one short sentence about cache locality." --max-new-tokens 8
```

## Main Benchmark Lanes

TinyLlama exact HF compare:

```bash
.venv/bin/python benchmarks/bench_llama_compare.py --backend torch_mps --device mps --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

SmolLM2 frontier sweep:

```bash
bash scripts/run_smollm2_frontier_compare.sh
```

Llama 3.2 3B stretch-model lane:

```bash
bash scripts/run_llama32_compare.sh
```

Qwen2.5 3B native-weight lane:

```bash
bash scripts/run_qwen25_compare.sh
```

GGUF / `llama.cpp` reference lanes:

```bash
bash scripts/run_llama32_gguf_reference.sh
bash scripts/run_qwen25_gguf_reference.sh
```

Shared model matrix:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --output-format pretty
```

Offline `vLLM` benchmark scaffold:

```bash
.venv/bin/python benchmarks/bench_vllm_offline.py --help
```

## Repository Layout

```text
dotcache/
  backends/         CPU, MPS, and CUDA backends
  integrations/     HF model adapters and vLLM scaffolding
  modes/            M0, M1, M2, M3, Turbo3
benchmarks/         Kernel, model, and runtime benchmarks
scripts/            Repeatable local benchmark wrappers
docs/               Benchmark summaries, journals, and roadmap notes
tests/              Unit and parity coverage
configs/            Local benchmark/config scaffolding
```

## Recommended Reading

- [`docs/benchmark_report.md`](./docs/benchmark_report.md) for the compact benchmark summary
- [`docs/performance_journal.md`](./docs/performance_journal.md) for the latest experiment log and CUDA checkpoints
- [`docs/model_roadmap.md`](./docs/model_roadmap.md) for target models and next lanes
- [`dotcache_software_implementation_guide.md`](./dotcache_software_implementation_guide.md) for implementation notes

## Summary

DotCache is no longer just a CPU/bootstrap prototype. It is a working exact compressed-domain KV-cache runtime with real HF model integration, real MPS/CUDA benchmark coverage, strong long-context memory wins, and early `vLLM` scaffolding.

The next step is not "make it exist." The next step is "keep the exact path correct while making the real-model decode kernels materially faster."
