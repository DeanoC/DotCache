# Qwen35 HIP Direct Tracking 2026-04-13

This directory is the first checkpoint bundle for the direct HIP tracking workflow.

## Attempted Runs

### 1. Direct HIP

Command:

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
  --record-json benchmarks/results/qwen35_hip_direct_tracking_20260413/short_prompt.json
```

Result:

- failed before inference
- error:

```text
direct HIP runtime requires a gfx11 HIP target, got host
```

### 2. Native HIP

Command:

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
  --load-mode native \
  --record-json benchmarks/results/qwen35_hip_direct_tracking_20260413/short_prompt_native.json
```

Result:

- failed during execution
- error:

```text
unsupported dtype BF16 for op matmul
```

### 3. CPU Direct

Command:

```bash
cargo run --manifest-path rust/Cargo.toml \
  -p dotcache-paged-runtime \
  --example hf_qwen35_minimal \
  --features qwen35-minimal,qwen35-minimal-hip \
  -- \
  'Qwen/Qwen3.5-0.8B' \
  'Hello from DotCache' \
  8 \
  --device cpu \
  --load-mode direct \
  --record-json benchmarks/results/qwen35_hip_direct_tracking_20260413/short_prompt_cpu_direct.json
```

Result:

- not completed
- runtime was not practical for interactive checkpointing on this host
- process was terminated after several minutes of sustained CPU use with no artifact yet written

## Current Tracking Status

The tracking harness is in place:

- canonical protocol: [qwen35_hip_direct_tracking.md](/home/deano/DotCache/docs/qwen35_hip_direct_tracking.md)
- machine-readable checkpoint output support: [hf_qwen35_minimal.rs](/home/deano/DotCache/rust/paged-runtime/examples/hf_qwen35_minimal.rs)

The remaining blocker is execution availability, not artifact format.

## Next Required Fixes

1. Fix direct HIP target detection so `hip-direct` no longer reports `got host`.
2. Fix the HIP BF16 matmul execution path on the active native lane.
3. Re-run the short prompt checkpoint and then add a longer decode checkpoint in this directory.
