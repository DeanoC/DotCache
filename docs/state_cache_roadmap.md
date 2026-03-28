# StateCache Roadmap

This note scopes the first DeltaNet-side follow-on to DotCache.

## Goal

Use Qwen3.5 as the first hybrid-family probe to understand whether recurrent DeltaNet state can support a DotCache-like compressed execution path.

The first milestone is intentionally limited to:

- dense-only Qwen3.5 DeltaNet state inspection
- dense-only recurrent-state ablations
- a synthetic StateCache simulator for `M0` and `M3`

It explicitly does **not** include:

- a compressed recurrent-state runtime
- multimodal/image support
- weight quantization
- a full hybrid-state abstraction that replaces native Qwen3.5 cache objects

## How It Relates To DotCache

DotCache still covers the attention-side KV path.

StateCache is the parallel exploration lane for the DeltaNet / `linear_attention` state family:

- DotCache: page-oriented compressed KV execution
- StateCache: tile-oriented compressed recurrent-state exploration

The reusable ideas are the policy and codec discipline:

- explicit mode signatures
- low-bit `M0`
- high-precision `M3` escape
- future policy gates around when to stay dense vs compress

The physical unit is different:

- attention path: KV pages
- DeltaNet path: recurrent-state tiles

## V1 Deliverables

- `bench_qwen35_deltanet_state_inspect.py`
  - measure where DeltaNet state bytes live
  - capture per-step conv and recurrent state deltas
- `bench_qwen35_deltanet_state_ablation.py`
  - compare pre-update, post-update, readout-only, and full-path perturbations
- `bench_state_cache_sim.py`
  - synthetic simulator for compression ratio and long-horizon error curves

## Deferred Follow-On

- real compressed DeltaNet runtime
- CUDA kernels for recurrent-state decode/update
- model-generic state abstractions beyond Qwen3.5
- learned policy selection

The intent is to answer the structural questions first on local hardware, then move runtime work to CUDA.
