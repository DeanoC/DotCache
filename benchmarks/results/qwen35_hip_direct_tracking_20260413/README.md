# Qwen35 HIP Direct Tracking 2026-04-13

Status: `hip-direct` is runnable on `hip:0` and the harness now distinguishes oracle choice.

Native-device oracle short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- oracle: `native-device`
- oracle_device: `hip`
- prefill_max_delta: `0.000000`
- decode_max_delta: `0.437500`

CPU oracle short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- oracle: `cpu`
- oracle_device: `cpu`
- prefill_max_delta: `4.710938`
- decode_max_delta: `5.808594`

Legacy longer CPU-oracle checkpoint:
- generated_text: `Hello from DotCache!

I!

I'm from`
- prompt_token_count: 4
- generated_token_count: 8
- device_load_ms: 3766.14
- device_prefill_ms: 1792.66
- device_decode_ms: 3178.06
- prefill_max_delta: 4.710938
- decode_max_delta: 6.414062

Current notes:
- CPU oracle still measures cross-device/backend drift and should not be used to judge direct-HIP correctness.
- Same-device tracing shows:
  - direct prefill logits match native HIP exactly
  - prefill cache tensors match native HIP exactly
  - decode input hidden state matches native HIP exactly
  - per-layer direct decode matches native HIP exactly when run from the same state
- the remaining same-device decode gap is smaller (`0.4375`) and is now narrowed to decode-step orchestration outside those traced subcomponents.
