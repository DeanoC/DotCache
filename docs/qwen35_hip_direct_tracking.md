# Qwen35 HIP Direct Tracking

This note defines the checkpoint format for the direct HIP lane while it is still moving quickly.

## Goal

Every meaningful direct-HIP change should leave behind:

- one correctness signal
- one profiling signal
- one machine-readable artifact

Ad hoc terminal readouts are no longer enough.

## Canonical Command

Use [hf_qwen35_minimal.rs](/home/deano/DotCache/rust/paged-runtime/examples/hf_qwen35_minimal.rs) for checkpoint runs because it already compares the device path against a reference path and reports timing deltas.

Example:

```bash
DOTCACHE_HIP_TRACE_CANDLE_FALLBACK=1 \
cargo run --manifest-path rust/Cargo.toml \
  -p dotcache-paged-runtime \
  --example hf_qwen35_minimal \
  --features qwen35-minimal,qwen35-minimal-hip \
  -- \
  'Qwen/Qwen3.5-0.8B' \
  'Hello from DotCache' \
  8 \
  --device hip:0 \
  --load-mode hip-direct \
  --record-json benchmarks/results/qwen35_hip_direct_latest.json
```

## Required Fields

The JSON artifact should capture at least:

- `model_id`
- `prompt`
- `device`
- `load_mode`
- `device_only`
- `prompt_token_count`
- `generated_token_count`
- `max_new_tokens`
- `cpu_load_ms`
- `device_load_ms`
- `cpu_prefill_ms`
- `device_prefill_ms`
- `cpu_decode_ms`
- `device_decode_ms`
- `prefill_max_delta`
- `decode_max_delta`
- `generated_text`
- `hip_trace_candle_fallback`
- `hip_print_transfers`
- `full_prefill_megakernel_requested`
- `hip_persistent_full_prefill_requested`

## Correctness Gates

Track these after each direct-HIP change:

- `prefill_max_delta`
- `decode_max_delta`
- generated text drift relative to the same prompt/load-mode checkpoint
- NaN warnings in prefill or decode

If deltas or output drift change, note whether the change is:

- expected numerical drift from a staging/layout rewrite
- an actual regression

## Profiling Gates

Track these after each direct-HIP change:

- `device_load_ms`
- `device_prefill_ms`
- `device_decode_ms`
- whether `DOTCACHE_HIP_TRACE_CANDLE_FALLBACK=1` emitted any fallback lines
- optional HIP transfer counters when `DOTCACHE_QWEN35_PRINT_HIP_TRANSFERS=1`

The direct lane should trend toward:

- flat or falling `device_decode_ms`
- no new fallback trace lines
- no growing host/device transfer footprint after load

## Update Rule

For every substantial direct-HIP execution change:

1. Run the canonical command on at least one short prompt.
2. Save the JSON artifact.
3. Record the commit and the main deltas in the PR or commit message if they changed materially.
4. If the change touches decode staging, prefer also running one longer decode case.

## Current Focus

The current high-value metrics are:

- decode latency
- correctness deltas against the reference path
- absence of Candle fallback traces on the live HIP path

That is a better signal than broad benchmark expansion while the direct lane is still being structurally rewritten.
