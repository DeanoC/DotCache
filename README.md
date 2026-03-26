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

## MPS Tuning Notes

The current eager `torch_mps` path is sensitive to page size.

Decode-step execution now batches compatible prepared pages on-device, so warm-cache performance is substantially better than the original per-page loop.
Preparation also batches compatible page uploads and keeps stored affine metadata compact on-device, so benchmarked `prepare_ms` and host-to-device bytes reflect the real page tensors rather than widened staging copies.

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
