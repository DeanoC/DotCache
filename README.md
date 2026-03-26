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

To sweep cache capacity under growing-context decode and compare FIFO vs LRU, use:

```bash
.venv/bin/python benchmarks/bench_decode_eviction.py --config configs/dotcache_m4_mps.yaml --contexts 4096 --decode-steps 8
```

That benchmark reports the tradeoff between:

- cache capacity in appended page-pairs
- eviction policy (`fifo` vs `lru`)
- access pattern (`all` history vs recent-window-heavy working set)
- hit rate and evictions
- per-step host-to-device bytes
- decode throughput versus CPU

Useful capacity labels:

- `initial`: enough resident space for the starting context only
- `final`: enough resident space for the fully grown context
- `unbounded`: no resident cap

To reproduce the crossover sweep:

```bash
bash scripts/run_mps_page_sweep.sh --config configs/dotcache_m4_mps.yaml
```

On this Mac setup, invoking the wrapper through `bash` is the most reliable path.
