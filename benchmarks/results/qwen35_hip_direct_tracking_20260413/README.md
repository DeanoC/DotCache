# Qwen35 HIP Direct Tracking 2026-04-13

Status: `hip-direct` is runnable on `hip:0`, the harness distinguishes oracle choice, and the same-device correctness drift is fixed.

Native-device oracle short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- oracle: `native-device`
- oracle_device: `hip`
- prefill_max_delta: `0.000000`
- decode_max_delta: `0.000000`
- device_load_ms: `1002.80`
- device_prefill_ms: `1799.17`
- device_decode_ms: `893.48`

Native-device oracle longer checkpoint:
- generated_text: `The direct HIP lane should stay correct while we keep specializing the decode path.
? (FAST) straight to the`
- prompt_token_count: 15
- generated_token_count: 8
- oracle: `native-device`
- oracle_device: `hip`
- prefill_max_delta: `0.000000`
- decode_max_delta: `0.000000`
- device_load_ms: `1020.97`
- device_prefill_ms: `4596.90`
- device_decode_ms: `7605.35`

CPU oracle short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- oracle: `cpu`
- oracle_device: `cpu`
- prefill_max_delta: `4.710938`
- decode_max_delta: `5.808594`
- device_load_ms: `3846.39`
- device_prefill_ms: `1799.83`
- device_decode_ms: `432.00`

Current notes:
- CPU oracle still measures cross-device/backend drift and should not be used to judge direct-HIP correctness.
- Same-device tracing proved:
  - direct prefill logits match native HIP exactly
  - prefill cache tensors match native HIP exactly
  - decode input hidden state matches native HIP exactly
  - per-layer direct decode matches native HIP exactly when run from the same state
  - whole-step direct decode matches native HIP exactly when run from the same state
- The remaining correctness bug was not direct-executor math. It was the old direct-runner env override bundle.
- Current baseline for `hip-direct` correctness work is:
  - `prefill_max_delta = 0.0`
  - `decode_max_delta = 0.0`
  - short and longer native-device checkpoints both green
