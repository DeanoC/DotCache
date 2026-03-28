# Qwen3.5 StateCache Showcase

This document is the compact external-facing summary for the current Qwen3.5 StateCache work.

It is intentionally narrower than [performance_journal.md](/workspace/DotCache/docs/performance_journal.md):

- one family: `Qwen3.5`
- one mechanism: DeltaNet `StateCache`
- one deployment story: native Hugging Face runtime on CUDA
- one scaling story: serving-only long-context runs

The current showable claim is:

`StateCache works on native HF Qwen3.5 up to 9B, reaches 32K context on a 32 GB GPU in serving mode, and materially improves decode throughput while compressing fixed resident DeltaNet state.`

## What Is Being Compressed

StateCache compresses the fixed resident `linear_attention` recurrent state in Qwen3.5.

It does **not** currently compress the token-growing `full_attention` KV state.

That means:

- fixed resident savings stay strong as prompt length grows
- total memory savings shrink at long context because attention KV becomes the dominant term

## Benchmark Modes

Two benchmark modes matter:

### Compare-mode

Use compare-mode when you need a clean before/after validation:

- dense baseline
- StateCache path
- same model
- same runtime
- same weights

This is the right mode for:

- greedy agreement
- short/medium prompt validation
- apples-to-apples throughput deltas

### Serving-mode

Use serving-mode when you need the real long-context scaling envelope:

- one dense prefill to capture native recurrent state
- convert recurrent state into StateCache form
- StateCache-only decode

This is the right mode for:

- prompt-length ceiling
- peak VRAM
- realistic StateCache deployment behavior

## Current Headline Results

### Compare-mode Highlights

These are the current short/medium-context proof points:

| Model | Prompt | Weights | Agreement | Dense tok/s | StateCache tok/s | Speedup | Fixed Saving |
|---|---:|---|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | `1024` | `fp16` | `1.00` | `24.32` | `61.65` | `2.53x` | `65.67%` |
| `Qwen/Qwen3.5-4B` | `1024` | `fp16` | `1.00` | `13.96` | `38.23` | `2.74x` | `58.33%` |
| `Qwen/Qwen3.5-9B` | `1024` | `bnb_8bit` | `1.00` | `4.30` | `10.32` | `2.40x` | `66.67%` |

Notes:

- `4B` uses the validated recurrent `M3` escapes on layers `0,1,2`
- `9B` is currently validated on the same native HF path with `bnb_8bit` weights

### Serving-mode Long-Context Highlights

These are the current long-context proof points:

| Model | Prompt | Weights | StateCache tok/s | Prefill Peak GB | Decode Peak GB | Fixed Saving | Total Saving | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.5-0.8B` | `32768` | `bnb_8bit` | `13.41` | `17.91` | `18.29` | `65.67%` | `3.07%` | `ok` |
| `Qwen/Qwen3.5-4B` | `32768` | `bnb_8bit` | `9.91` | `22.77` | `23.80` | `58.33%` | `2.69%` | `ok` |
| `Qwen/Qwen3.5-9B` | `32768` | `bnb_8bit` | `9.64` | `29.14` | `30.07` | `66.67%` | `3.07%` | `ok` |

Current ceiling on this pod:

- exact `32768` passes for `0.8B`, `4B`, and `9B`
- exact `65536` still fails

The `65536` failures are dominated by a single large long-context allocation spike rather than recurrent-state storage.

## What This Proves

This is enough to show:

- StateCache is not just a toy `0.8B` result
- it survives a realistic quantized-weight deployment setup
- it works on the native HF runtime rather than only a custom inference stack
- it improves decode throughput while materially reducing fixed resident state
- it reaches `32K` on a `32 GB` GPU at least through `Qwen3.5-9B`

## What It Does Not Yet Prove

It does **not** yet prove:

- end-to-end superiority versus external KV-cache quant runtimes like TurboQuant or GGUF systems
- total-memory dominance at long context
- support for `65536+` on this pod
- a polished production serving stack beyond the current benchmark harnesses

## How To Reproduce

### Compare-mode

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 512 1024 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --continue-on-error
```

### Serving-mode

```bash
source scripts/env_cuda.sh
.venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_serving.py \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --weight-quantization bnb_8bit \
  --target-prompt-lengths 16384 32768 \
  --max-new-tokens 2 \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --continue-on-error
```

### Render A Compact Table From JSONL

```bash
python scripts/report_qwen35_statecache_showcase.py \
  --input /path/to/readout.jsonl \
  --input /path/to/serving.jsonl
```

## How To Talk About External Systems

If you want to compare this against:

- GGUF
- TurboQuant
- other KV-cache quant runtimes

keep that as a separate appendix.

Reason:

- StateCache compresses recurrent state inside native Qwen3.5 HF execution
- KV-cache quant systems usually compress a different state family in a different runtime

So the fair message is:

- `StateCache vs dense HF` is the direct apples-to-apples claim
- `StateCache vs TurboQuant/GGUF` is a deployment-envelope comparison, not a like-for-like mechanism comparison

## Next External-Facing Step

The next best way to make this easier to show is:

1. rerun one locked benchmark set cleanly
2. save the raw JSONL in `benchmarks/results/`
3. render the markdown table with `scripts/report_qwen35_statecache_showcase.py`
4. optionally add one throughput plot and one peak-VRAM plot
