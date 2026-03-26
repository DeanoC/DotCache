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

