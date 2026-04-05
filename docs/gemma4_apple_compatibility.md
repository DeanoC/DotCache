# Gemma 4 Apple Compatibility

This note captures the current Gemma 4 status on the local Apple Silicon development path.

Observed on April 5, 2026 on an Apple Silicon Mac with 16 GB unified memory.

## Compatibility Matrix

| Lane | Model | Prompt Tokens | Decode Tokens | Status | Notes |
| --- | --- | ---: | ---: | --- | --- |
| Dense HF on MPS | `google/gemma-4-E2B` | 7 | 1 | Fails | `RuntimeError: Invalid buffer size: 9.54 GiB` during model load on the plain dense path. |
| DotCache on MPS (`balanced`) | `google/gemma-4-E2B` | 7 | 1 | Works | Completed in about `99s`, produced the same token as the dense teacher-forced path, and reported `greedy_token_agreement_rate=1.0`. |
| Tiny random Gemma 4 on `torch_mps` | synthetic config | 4 | 4 | Works | Harness-level Apple smoke passed with exact greedy-token agreement and low logit drift. |

## Reproduce

Run the Apple smoke lane:

```bash
bash scripts/run_gemma4_apple_smoke.sh
```

That smoke writes:

- `dotcache_mps_balanced.json`: the raw probe result
- `smoke_runner.json`: the wrapper-level timeout/exit summary

Run the short DotCache-only Apple benchmark lane:

```bash
bash scripts/run_gemma4_mps_short_bench.sh
```

If you want to inspect the plain dense failure mode directly, run:

```bash
.venv/bin/python scripts/probe_gemma4_text.py \
  --model-id google/gemma-4-E2B \
  --device-map mps \
  --torch-dtype bfloat16 \
  --max-new-tokens 1 \
  --prompt "Cache locality on Apple Silicon is"
```

The Apple smoke wrapper intentionally uses the DotCache path because that is the lane that fits and completes on this machine.
