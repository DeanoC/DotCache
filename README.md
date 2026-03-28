# DotCache

This repository is a software-first prototype of DotCache: executing decode-time attention directly on compressed KV-cache pages.

The current bootstrap intentionally focuses on the boring, load-bearing pieces first:

- a readable CPU reference path
- `M0` affine and symmetric grouped quantization
- `M3` high-precision escape pages
- Selective Key Precision via model-specific sensitivity maps
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
- [docs/local_layer_profiles.md](./docs/local_layer_profiles.md)
- [docs/cuda_next_steps.md](./docs/cuda_next_steps.md)
- [docs/turboquant_comparison_plan.md](./docs/turboquant_comparison_plan.md)
- [docs/state_cache_roadmap.md](./docs/state_cache_roadmap.md)

## Quick start on NVIDIA Linux / 5090-class CUDA pods

This repo now has a `torch_cuda` backend for NVIDIA development, so the practical CUDA path today is:

- run both the dense Hugging Face model and DotCache exact decode on `cuda`
- use the CUDA unit tests plus the HF compare harnesses to verify parity on this machine
- use the 5090 pod as the first larger-model HF scale-up lane
- use the optional vLLM adapter only when you are ready for the Phase 6 offline benchmark path

Bootstrap the local environment with:

```bash
source scripts/env_cuda.sh
bash scripts/bootstrap_nvidia_llama_dev.sh
```

The sourced env script normalizes `CUDA_HOME`, `CUDA_PATH`, `PATH`, `LD_LIBRARY_PATH`, and a shared Hugging Face cache under `/workspace/.cache/huggingface`, so a pod move does not depend on whatever the new login shell happens to export.

The bootstrap script then creates `.venv`, reuses a working system CUDA PyTorch when the pod already has one, otherwise installs a current `torch>=2.8` wheel, then installs the dev and Hugging Face dependencies and fails fast if `torch.cuda.is_available()` is false.

For gated Hugging Face checkpoints such as Llama 3.2, export your token before running the model benchmarks:

```bash
export HF_TOKEN=...
source scripts/env_cuda.sh
```

The env script mirrors `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`, and the current harnesses / benchmark entrypoints now pass that token into `from_pretrained(...)`.

After a pod move, the minimal recovery flow is:

```bash
cd /workspace/DotCache
source scripts/env_cuda.sh
bash scripts/bootstrap_nvidia_llama_dev.sh
```

For a local no-download smoke run on this machine, use:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend cpu_ref --device cuda --max-new-tokens 4
```

For a real checkpoint on NVIDIA, start with:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --backend cpu_ref --device cuda --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Write one short sentence about cache locality." --max-new-tokens 8
```

For the 5090-era HF scale-up lane on this pod, start with:

```bash
bash scripts/run_qwen25_compare_cuda.sh
bash scripts/run_qwen25_7b_compare_cuda.sh
bash scripts/run_llama32_compare_cuda.sh
```

On Qwen2.5 CUDA lanes, those public wrappers now default to `K=M3 / V=M0`. That is the current recommended path on the 5090 pod because `M0/M0` loses agreement at `1024/2048` while the key-exact lane restores parity.

DotCache now also has a more surgical capability behind the same runtime: Selective Key Precision. A model-specific Sensitivity Map can keep exact key pages only in fragile layers or KV groups while the rest of the cache stays compressed.

The companion reporting artifact is now the compressibility profile:

```bash
.venv/bin/python scripts/report_compressibility_profiles.py --backend torch_cuda
```

That report summarizes, per model and prompt length:

- whether the model tolerates all-`M0`
- whether it benefits from Selective Key Precision
- what fraction of K pages must stay exact
- the KV-memory tradeoff versus full exact-K
- the observed decode throughput for each recorded policy

For automated sensitivity-map suggestions on a single model/prompt, use:

```bash
.venv/bin/python scripts/suggest_selective_k_policy.py \
  --family qwen2 \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --prompt-length 2048
```

That tool runs one offline fidelity capture, scores selective exact-K candidates, and emits a budget-aware recommended policy plus a ready-to-run benchmark command.

To batch that validated search across the current public CUDA model set and emit one compact table, use:

```bash
.venv/bin/python scripts/build_compressibility_map.py --backend torch_cuda --device cuda
```

That batch report validates `all M0`, validates the top selective candidates, and emits a compact per-model row with the chosen policy, exact-K fraction, KV-memory ratio, and decode throughput.

For local Apple MPS investigation of the revised paper's layer/page policy idea, see the first-pass handwritten profiles and probe tools in:

- [local_layer_profiles.md](/Users/deanocalver/Documents/Projects/DotCache/docs/local_layer_profiles.md)
- [bench_layer_sensitivity.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_layer_sensitivity.py)
- [inspect_policy_prefill.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/inspect_policy_prefill.py)

For a narrow high-context CUDA slice that fits this pod cleanly, use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/build_compressibility_map.py \
  --spec 'qwen2|Qwen/Qwen2.5-3B-Instruct|4096' \
  --spec 'llama|HuggingFaceTB/SmolLM2-360M-Instruct|4096' \
  --backend torch_cuda \
  --device cuda
```

That is the current practical route to a `4096` public-model data point here: Qwen2.5 3B selective plus a SmolLM2 360M tolerant reference.

And summarize recorded benchmark history with:

```bash
.venv/bin/python scripts/report_model_benchmarks.py --benchmark qwen2_compare
```

Canonical Qwen CUDA record labels on the 5090 pod:

```bash
.venv/bin/python scripts/record_benchmark.py --label qwen25-3b-cuda-k-exact -- bash scripts/run_qwen25_compare_cuda.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-3b-cuda-selective -- bash scripts/run_qwen25_compare_cuda_selective.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda-k-exact -- bash scripts/run_qwen25_7b_compare_cuda.sh
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda-selective -- bash scripts/run_qwen25_7b_compare_cuda_selective.sh
```

For the old `M0/M0` comparison lane, override the wrapper defaults:

```bash
.venv/bin/python scripts/record_benchmark.py --label qwen25-3b-cuda -- bash scripts/run_qwen25_compare_cuda.sh --default-mode-k M0 --default-mode-v M0
.venv/bin/python scripts/record_benchmark.py --label qwen25-7b-cuda -- bash scripts/run_qwen25_7b_compare_cuda.sh --default-mode-k M0 --default-mode-v M0
```

The first validated selective policy is Qwen2.5 3B on CUDA:

- exact K on `layer 0`
- exact K on `layer 27`, `KV1`
- `M0` for all other key pages
- `M0` for all value pages

Run that lane with:

```bash
bash scripts/run_qwen25_compare_cuda_selective.sh
bash scripts/run_qwen25_7b_compare_cuda_selective.sh
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
bash scripts/run_qwen25_compare.sh
```

9. External GGUF / llama.cpp reference lane:

```bash
bash scripts/run_llama32_gguf_reference.sh
bash scripts/run_qwen25_gguf_reference.sh
```

10. Optional mounted-HF fetch lane via `hf-mount`:

```bash
bash scripts/run_llama32_compare_mounted.sh
bash scripts/run_qwen25_compare_mounted.sh
```

11. CUDA scale-up lane on a large NVIDIA pod:

```bash
bash scripts/run_qwen25_compare_cuda.sh
bash scripts/run_qwen25_7b_compare_cuda.sh
bash scripts/run_llama32_compare_cuda.sh
```

The Qwen CUDA wrappers in that lane intentionally default to `K=M3 / V=M0` today. Keep `M0/M0` as the baseline for Llama/SmolLM lanes.

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

For the first non-Llama native-weight target on the same HF path, use Qwen2.5 3B:

```bash
bash scripts/run_qwen25_compare.sh
```

That wrapper targets `Qwen/Qwen2.5-3B-Instruct` through the new Qwen2-specific attention adapter in [qwen2.py](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/qwen2.py), but reuses the same replay/generation/loss harness shape as the existing Llama path. It is treated as a stretch-model lane on this Mac and defaults to exact prompt lengths `1024 2048` with `--continue-on-error`.

For the first runnable Qwen3.5 hybrid-family lane, use the new text-only dense smoke harness:

```bash
.venv/bin/python benchmarks/bench_qwen35_text.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --max-new-tokens 2 --target-prompt-lengths 512
```

That lane is intentionally dense-only and text-only. It proves the shared model matrix and benchmark surface can handle Qwen3.5 without pretending DotCache already supports the hybrid attention/delta state path.

To inspect where DotCache could attach later, use the hybrid-state inspection runner:

```bash
.venv/bin/python benchmarks/bench_qwen35_hybrid_inspect.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --target-prompt-lengths 128
```

That runner reports:
- which text layers are `full_attention` vs `linear_attention`
- how much prefill state lives in attention KV vs convolution/recurrent state
- whether an attention-subset-only DotCache path is a coherent next step or whether a broader hybrid-state abstraction is required

For the next step after inspection, there is also a dense-only attention-subset capture runner:

```bash
.venv/bin/python benchmarks/bench_qwen35_attention_subset.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --target-prompt-lengths 128 --max-new-tokens 2
```

That runner only wraps the `full_attention` layers. It leaves every `linear_attention` / DeltaNet layer on the native dense path, and records decode-time Q/K/V/context for the attention subset so we can prototype partial DotCache support without pretending the recurrent state problem is solved.

There is now also an attention-subset DotCache replay runner for the same six `full_attention` layers:

```bash
.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --target-prompt-lengths 64 --max-new-tokens 2 --tokens-per-page 16
```

That lane seeds DotCache only from the native attention KV cache, leaves every DeltaNet / `linear_attention` layer on the native hybrid cache path, and measures replay/logit drift for the attention subset. It is the first partial DotCache integration point for Qwen3.5, but it is still not full hybrid-state support.

There is now also a layer-aware profile family for that lane:

```bash
.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --repeat-counts --target-prompt-lengths 32 --max-new-tokens 1 --tokens-per-page 16 --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_second_pass.yaml
```

For the parallel DeltaNet-side probe lane, use the new StateCache inspection and ablation runners:

```bash
.venv/bin/python benchmarks/bench_qwen35_deltanet_state_inspect.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --target-prompt-lengths 32 --max-new-tokens 2
.venv/bin/python benchmarks/bench_qwen35_deltanet_state_ablation.py --model-id Qwen/Qwen3.5-0.8B --backend torch_mps --device mps --target-prompt-lengths 32 --max-new-tokens 2 --bits 8 4
.venv/bin/python benchmarks/bench_state_cache_sim.py --state-rows 128 --state-cols 128 --steps 16 --bits 8 4 3 --modes M0 M3
```

To bridge real Qwen3.5 state into the simulator and sweep early/mid/late layers in one pass, use:

```bash
.venv/bin/python benchmarks/bench_qwen35_statecache_real_sweep.py --backend torch_mps --device mps --prompt-length 32 --max-new-tokens 4 --layers 0 12 22 --state-kinds recurrent conv
```

Those lanes are intentionally probe-only:

- they inspect and perturb DeltaNet recurrent state
- they do not implement a compressed recurrent-state runtime
- they use `M0` low-bit and `M3` escape as the first codec pair
- they are meant to guide the CUDA-side StateCache work, not replace the existing attention-side DotCache lane
- the real-sweep wrapper now emits per-layer recommendation records so recurrent and conv state can be compared side by side

That profile only applies to the six `full_attention` layers, disables the recent-window escape so the probe actually hits sealed static pages, and keeps the DeltaNet / `linear_attention` state on the native path. The safer second pass uses explicit `M0`-first value overrides for the fragile late attention layers instead of relying on generic value `strict` tiering.

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

If you want to avoid fully downloading large HF repos first, there is now an optional `hf-mount` lane:

```bash
bash scripts/run_llama32_compare_mounted.sh
bash scripts/run_qwen25_compare_mounted.sh
```

Those wrappers call [bench_hf_mount_compare.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_hf_mount_compare.py), which:

- probes `hf-mount`
- mounts the target HF repo as a local filesystem
- runs the existing HF compare harness against the mounted path
- stops the mount afterward unless you ask to keep it

You can also ask the shared matrix to emit mounted-HF commands instead of direct Hub loads:

```bash
.venv/bin/python benchmarks/bench_model_matrix.py --model-keys llama32_3b_hf qwen25_3b_hf --mount-hf-models --output-format pretty
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
