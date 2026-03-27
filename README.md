# DotCache

This repository is a software-first prototype of DotCache: executing decode-time attention directly on compressed KV-cache pages.

The current bootstrap intentionally focuses on the boring, load-bearing pieces first:

- a readable CPU reference path
- `M0` affine and symmetric grouped quantization
- `M3` high-precision escape pages
- page packing and metadata round-trips
- streaming `score_page` and `mix_page` that avoid full-page materialization by default
- tests that compare compressed-domain execution against an explicit dequantized baseline built from the same quantized pages

## Reference docs

- [dotcache_full.tex](./dotcache_full.tex)
- [dotcache_software_implementation_guide.md](./dotcache_software_implementation_guide.md)
- [dotcache_no_cuda_bootstrap_m4_amd.md](./dotcache_no_cuda_bootstrap_m4_amd.md)
- [dotcache_nvidia_llama_bootstrap.md](./dotcache_nvidia_llama_bootstrap.md)
- [docs/benchmark_report.md](./docs/benchmark_report.md)
- [docs/performance_journal.md](./docs/performance_journal.md)
- [docs/model_roadmap.md](./docs/model_roadmap.md)
- [docs/turboquant_comparison_plan.md](./docs/turboquant_comparison_plan.md)

## Quick start on NVIDIA Linux for non-MPS Llama work

This repo now has a `torch_cuda` backend for NVIDIA development, so the practical CUDA path today is:

- run both the dense Hugging Face model and DotCache decode on `cuda`
- use the CUDA unit tests plus the Llama harness to verify parity on this machine
- use the optional vLLM adapter only when you are ready for the Phase 6 offline benchmark path

Bootstrap the local environment with:

```bash
bash scripts/bootstrap_nvidia_llama_dev.sh
```

That script creates `.venv`, reuses a working system CUDA PyTorch when the pod already has one, otherwise installs the pinned driver-compatible wheel, then installs the dev and Hugging Face dependencies and fails fast if `torch.cuda.is_available()` is false.

For a local no-download smoke run on this machine, use:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend cpu_ref --device cuda --max-new-tokens 4
```

For a real checkpoint on NVIDIA, start with:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --backend cpu_ref --device cuda --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Write one short sentence about cache locality." --max-new-tokens 8
```

## Quick start on Apple Silicon

1. Install Python 3.11+.
2. Create a virtualenv:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

3. Run the tests:

```bash
./scripts/run_unit_tests.sh
```

4. Optional MPS dependency for the next stage:

```bash
.venv/bin/pip install -e ".[dev,mps]"
```

5. Optional Hugging Face dependency for the Phase 5 model-integration path:

```bash
.venv/bin/pip install -e ".[dev,mps,hf]"
```

6. Optional vLLM dependency for the Phase 6 CUDA runtime-integration path:

```bash
.venv/bin/pip install -e ".[dev,hf,vllm]"
```

7. Dedicated local Turbo3 smoke lane on MPS:

```bash
bash scripts/run_turbo3_mps_suite.sh tinyllama
bash scripts/run_turbo3_mps_suite.sh smollm2
```

8. Stretch-model local HF lane on MPS:

```bash
bash scripts/run_llama32_compare.sh
```

9. External GGUF / llama.cpp reference lane:

```bash
bash scripts/run_llama32_gguf_reference.sh
bash scripts/run_qwen25_gguf_reference.sh
```

## Current package layout

```text
configs/
dotcache/
  backends/
  modes/
tests/
benchmarks/
scripts/
```

## Status

This is the CPU-reference bootstrap, not the final runtime. The next logical step on this M4 Mac is a `torch_mps` execution backend that reuses the same page format and correctness harness.

## Phase 5 Llama Integration

The repo now includes a narrow Phase 5 model-integration path in [llama.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/llama.py):

- one Llama-family architecture path only
- dense prefill
- exact full-context DotCache decode only
- batch=1 greedy generation only
- no `generate()` patching, beam search, sampling, or vLLM integration in this phase

The public model-facing bridge is [model_kv_cache.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/model_kv_cache.py). It keeps per-layer, per-KV-head exact sessions and adds a tail-page builder so token-by-token append does not degenerate into persistent one-token pages.

For a local no-download smoke benchmark, use:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend cpu_ref --device cpu --max-new-tokens 4
```

On this M4, the same harness can also run on MPS:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend torch_mps --device mps --max-new-tokens 3
```

For the intended real-model path, the benchmark defaults to TinyLlama:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --backend torch_mps --device mps --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Write one short sentence about cache locality." --max-new-tokens 8
```

On an NVIDIA Linux box, use the same harness with `--device cuda --backend cpu_ref` until the CUDA DotCache backend exists:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --backend cpu_ref --device cuda --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Write one short sentence about cache locality." --max-new-tokens 8
```

That benchmark reports:

- prompt length
- dense prefill time
- prefill-cache ingest time
- per-step decode time
- internal append/decode runtime split inside the DotCache path
- host-to-device bytes
- resident bytes
- greedy token agreement versus the dense path
- teacher-forced logit drift versus the dense path

For a small higher-context Llama-family checkpoint on this M4, use SmolLM2 360M:

```bash
bash scripts/run_smollm2_long_context_compare.sh
```

That wrapper runs [bench_llama_compare.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_llama_compare.py) against `HuggingFaceTB/SmolLM2-360M-Instruct` at an exact `2048`-token prompt so we can exercise the Phase 5 path beyond TinyLlama's `2048`-token context ceiling. Exact `3072` and `4096` probes hit dense MPS OOM on this machine.

For the higher-context exact-length frontier on the same model, use:

```bash
bash scripts/run_smollm2_frontier_compare.sh
```

That runner sweeps exact prompt lengths `256 512 1024 1536 2048` from one model load. On the current M4 checkpoint, DotCache is still slower than dense through `1536` tokens in the one-load sweep, but it already uses much less KV memory, and a fresh standalone `2048` rerun shows DotCache ahead on decode while keeping the same KV-memory win.

For the next "proper model" target on the same HF path, use Llama 3.2 3B:

```bash
bash scripts/run_llama32_compare.sh
```

That wrapper targets `meta-llama/Llama-3.2-3B-Instruct` with exact prompt lengths `1024 2048` and `--continue-on-error`, so it behaves like a real stretch-model lane on this Mac instead of assuming every longer prompt will fit. The same target is also exposed through the shared model matrix:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf --output-format pretty
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf --run-supported --backend torch_mps --device mps
```

For the external GGUF / `llama.cpp` reference lane, use:

```bash
bash scripts/run_llama32_gguf_reference.sh
bash scripts/run_qwen25_gguf_reference.sh
```

Those wrappers call [bench_gguf_external.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_gguf_external.py), which:

- builds exact-length prompts with the matching Hugging Face tokenizer
- runs `llama-cli -hf <repo>`
- parses `llama.cpp` timing lines when they are available
- emits a clean error record instead of crashing if `llama-cli` is not installed

The same reference lanes are also exposed through the shared model matrix:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_gguf qwen25_3b_gguf --output-format pretty
```

## Phase 6 vLLM Integration

The repo now also has a correctness-first Phase 6 adapter surface in [dotcache/integrations/vllm_adapter](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/vllm_adapter):

- Llama-family only
- CUDA-only in the intended runtime path
- exact full-context decode only
- `dense`, `dotcache_shadow`, and `dotcache_active` modes
- `tokens_per_page == block_size` enforced as a hard invariant
- offline-engine benchmarking only for the first milestone

The real vLLM hook targets the pinned `0.18.x` line and is intentionally conservative about unknown versions.
Because `vllm 0.18.x` defaults `vllm.LLM` to a detached engine-core process, DotCache's current adapter path requires the in-process engine. Use `configure_vllm_inprocess_runtime()` before constructing `vllm.LLM`, or set `VLLM_ENABLE_V1_MULTIPROCESSING=0` yourself.

```python
from dotcache.integrations.vllm_adapter import configure_vllm_inprocess_runtime

configure_vllm_inprocess_runtime()

from vllm import LLM
```

For the new offline benchmark harness on a CUDA box with vLLM installed, use:

```bash
.venv-vllm/bin/python benchmarks/bench_vllm_offline.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --backend torch_cuda --block-size 16 --mode all --prompt-repeat-counts 1 8 32 --max-new-tokens 16
```

That benchmark prints dense, shadow, and active records with:

- block size
- prompt repeat count / tokenized prompt length
- decode steps
- wall-clock decode ms per step
- DotCache block-encode / append / decode runtime totals
- resident KV bytes
- host-to-device bytes
- greedy agreement versus dense when both paths are run

The adapter and benchmark surface are implemented and unit-tested locally, but the real vLLM CUDA numbers still need to be collected on the cloud instance.

## MPS Tuning Notes

The current eager `torch_mps` path is sensitive to page size.

Decode-step execution now batches compatible prepared pages on-device, so warm-cache performance is substantially better than the original per-page loop.
Preparation also batches compatible page uploads and keeps stored affine metadata compact on-device, so benchmarked `prepare_ms` and host-to-device bytes reflect the real page tensors rather than widened staging copies.
Runtime sketches for page-gating experiments are now computed at encode time, so session preload/append measurements no longer absorb on-the-fly sketch generation cost.

- With the current M4-tuned unpack path, `torch_mps` already wins over `cpu_ref` at long context for `tokens_per_page=64`.
- Larger pages still matter a lot because they let MPS amortize per-page overhead much better.
- On this M4 Mac, `tokens_per_page=256` is a strong default for MPS experiments, and `512` can be significantly faster again when the runtime can tolerate fewer, larger pages.

For a ready-made MPS-oriented profile, start from [configs/dotcache_m4_mps.yaml](./configs/dotcache_m4_mps.yaml).

Benchmark scripts accept `--config <path>` and then let explicit CLI flags override the loaded values. For example:

```bash
.venv/bin/python benchmarks/bench_decode.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096
```

To measure repeated decode steps with runtime page reuse, use:

```bash
.venv/bin/python benchmarks/bench_decode_reuse.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8
```

That benchmark reports:

- `no_cache_*`: re-preparing pages on every decode step
- `cache_cold_*`: one cold cache fill amortized across repeated steps
- `cache_warm_*`: steady-state decode with a warm prepared-page cache

To measure growing-context decode where only newly appended pages are prepared, use:

```bash
.venv/bin/python benchmarks/bench_decode_growth.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8
```

That benchmark models a resident prepared-page cache, appends one page of fresh KV per decode step, and reports how much host-to-device work remains per step once old pages stay warm.

To benchmark a more model-shaped runtime with distinct preload, append, and decode phases, use:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8
```

That benchmark keeps a resident session object alive across steps and reports:

- one-time preload latency and bytes
- per-step append latency and bytes
- per-step decode latency with resident pages
- combined session runtime per generated step

To evaluate a sink-plus-recent execution policy against the full-context oracle, add execution windows explicitly:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8 --execution-sink-window 256 --execution-recent-window 1024
```

That reports active page/token counts and numerical error versus the full CPU reference, so you can see the speed/accuracy tradeoff directly.
This policy is intentionally approximate in the current prototype; aggressive windows can cut decode cost sharply, but they can also introduce large output error versus full-context attention.

To recover a few older pages by cheap query relevance on top of sink-plus-recent, add `--execution-relevance-top-k`:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8 --execution-sink-window 256 --execution-recent-window 1024 --execution-relevance-top-k 4
```

This keeps the window policy as the base set, then admits a small number of older key/value page pairs whose page-summary vectors score highest against the current query.

To make that first-pass signal less blunt, raise `--execution-relevance-sketch-size` above `1` so each page is represented by several sub-page mean vectors instead of one global mean:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8 --execution-sink-window 256 --execution-recent-window 1024 --execution-relevance-top-k 4 --execution-relevance-sketch-size 4
```

To switch from sketch-based relevance to a stronger page-envelope score, use `--execution-relevance-mode envelope`:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8 --execution-sink-window 256 --execution-recent-window 1024 --execution-relevance-top-k 4 --execution-relevance-mode envelope
```

This uses encode-time per-page min/max envelopes to form a query-dependent upper-bound style score for each old page. On the current M4 prototype, that envelope gate is materially better than the sketch gate at roughly the same latency budget.
The best current M4 balance from our targeted sweep is `--execution-sink-window 256 --execution-recent-window 1024 --execution-relevance-top-k 4 --execution-relevance-mode envelope`.

To run that tuned M4 approximate profile directly, use:

```bash
bash scripts/run_m4_envelope_session.sh --contexts 4096
```

To run the faster fixed variant, use:

```bash
bash scripts/run_m4_envelope_fast_session.sh --contexts 4096
```

There is also an experimental context-aware variant:

```bash
bash scripts/run_m4_envelope_autoscaled_session.sh --contexts 4096 8192 16384
```

That profile scales the recent window and `top_k` with context length, but current validation says it is not yet a clear win over the simpler fixed `256/1024/4` profile.

To sweep that envelope profile around different `sink/recent/top_k` settings, use:

```bash
bash scripts/run_envelope_sweep.sh --config configs/dotcache_m4_mps.yaml --contexts 4096 --execution-sink-windows 128 256 384 --execution-recent-windows 768 1024 1280 --execution-relevance-top-ks 2 4 6
```

That emits JSONL sweep records, tags Pareto-frontier points, and can be redirected into a file such as [envelope_sweep_4k.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/envelope_sweep_4k.jsonl).
At longer contexts like `8192` and `16384`, the same fixed `256/1024/4` profile keeps latency almost flat but max-abs error rises, so context-scaled tuning is still the next step rather than treating one window as universal.

To tune `fast` and `balanced` long-context candidates under a runtime budget, use:

```bash
.venv/bin/python benchmarks/bench_decode_envelope_tuner.py --config configs/dotcache_m4_mps.yaml --contexts 8192 16384 > benchmarks/results/envelope_tuner_8k_16k.jsonl
```

That emits all candidate rows plus a summary row per context. The latest committed tuner output is [envelope_tuner_8k_16k.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/envelope_tuner_8k_16k.jsonl).

To refine that sketch shortlist with exact compressed-domain key scoring before final decode, add `--execution-exact-refine-top-k`:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8 --execution-sink-window 256 --execution-recent-window 1024 --execution-relevance-top-k 8 --execution-relevance-sketch-size 4 --execution-exact-refine-top-k 4
```

This is currently an experimental middle ground: it keeps sink and recent pages, admits a larger candidate pool from the chosen relevance mode, then uses exact page scoring to keep only the best old pages for the final decode. The current implementation reuses those exact shortlisted logits during final decode so it does not rescore the chosen old pages, but on the M4 prototype it is still much slower than the new envelope-only gate.

To approximate dropped old pages instead of ignoring them entirely, add `--execution-approximate-old-pages`:

```bash
.venv/bin/python benchmarks/bench_decode_session.py --backend torch_mps --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8 --execution-sink-window 256 --execution-recent-window 1024 --execution-approximate-old-pages
```

This uses exact decode on the active page set and a summary-based fallback contribution for older pages that stay outside the exact path.

To sweep cache capacity under growing-context decode and compare FIFO, LRU, and newest-page pinning, use:

```bash
.venv/bin/python benchmarks/bench_decode_eviction.py --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8
```

That benchmark reports the tradeoff between:

- cache capacity in appended page-pairs
- eviction policy (`fifo`, `lru`, or `pinned_recent_fifo`)
- optional newest-page pinning depth for the pinned policy
- access pattern (`all` history vs recent-window-heavy working set)
- hit rate and evictions
- per-step host-to-device bytes
- decode throughput versus CPU

Useful capacity labels:

- `initial`: enough resident space for the starting context only
- `final`: enough resident space for the fully grown context
- `unbounded`: no resident cap

To isolate the workload-shaped policy, add `--cache-policies pinned_recent_fifo --pinned-recent-page-pairs 4`.

To reproduce the crossover sweep:

```bash
bash scripts/run_mps_page_sweep.sh --config configs/dotcache_m4_mps.yaml
```

On this Mac setup, invoking the wrapper through `bash` is the most reliable path.
