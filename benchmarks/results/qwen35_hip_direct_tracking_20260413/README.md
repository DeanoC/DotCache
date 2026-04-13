# Qwen35 HIP Direct Tracking 2026-04-13

Status: `hip-direct` is runnable on `hip:0`, the harness distinguishes oracle choice, and the same-device correctness drift is fixed.

Native-device oracle short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- oracle: `native-device`
- oracle_device: `hip`
- prefill_max_delta: `0.000000`
- prefill_cache_max_delta: `0.000000`
- decode_max_delta: `0.000000`
- decode_input_hidden_max_delta: `0.000000`
- decode_step_cache_max_delta: `0.000000`
- device_load_ms: `972.03`
- device_prefill_ms: `1801.84`
- device_decode_ms: `949.14`

Native-device oracle longer checkpoint:
- generated_text: `The direct HIP lane should stay correct while we keep specializing the decode path.
? (FAST) straight to the`
- prompt_token_count: 15
- generated_token_count: 8
- oracle: `native-device`
- oracle_device: `hip`
- prefill_max_delta: `0.000000`
- prefill_cache_max_delta: `0.000000`
- decode_max_delta: `0.000000`
- decode_input_hidden_max_delta: `0.000000`
- decode_step_cache_max_delta: `0.000000`
- device_load_ms: `1009.79`
- device_prefill_ms: `4853.36`
- device_decode_ms: `7621.18`

CPU oracle short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- oracle: `cpu`
- oracle_device: `cpu`
- prefill_max_delta: `4.710938`
- prefill_cache_max_delta: `4.156250`
- decode_max_delta: `5.871094`
- decode_input_hidden_max_delta: `0.000000`
- decode_step_cache_max_delta: `4.156250`
- device_load_ms: `3701.04`
- device_prefill_ms: `1750.00`
- device_decode_ms: `959.67`

Recent history:
- 2026-04-13T14:05:05.303073+00:00 short_native_device prefill_max_delta=0.000000 prefill_cache_max_delta=null decode_max_delta=0.000000 decode_input_hidden_max_delta=null decode_step_cache_max_delta=null device_prefill_ms=1782.43 device_decode_ms=948.41
- 2026-04-13T14:05:36.431131+00:00 longer_native_device prefill_max_delta=0.000000 prefill_cache_max_delta=null decode_max_delta=0.000000 decode_input_hidden_max_delta=null decode_step_cache_max_delta=null device_prefill_ms=4745.81 device_decode_ms=7636.00
- 2026-04-13T14:06:42.474369+00:00 short_cpu_oracle prefill_max_delta=4.710938 prefill_cache_max_delta=null decode_max_delta=5.871094 decode_input_hidden_max_delta=null decode_step_cache_max_delta=null device_prefill_ms=1640.04 device_decode_ms=972.43
- 2026-04-13T14:11:18.916082+00:00 short_native_device prefill_max_delta=0.000000 prefill_cache_max_delta=0.000000 decode_max_delta=0.000000 decode_input_hidden_max_delta=0.000000 decode_step_cache_max_delta=0.000000 device_prefill_ms=1801.84 device_decode_ms=949.14
- 2026-04-13T14:11:51.272819+00:00 longer_native_device prefill_max_delta=0.000000 prefill_cache_max_delta=0.000000 decode_max_delta=0.000000 decode_input_hidden_max_delta=0.000000 decode_step_cache_max_delta=0.000000 device_prefill_ms=4853.36 device_decode_ms=7621.18
- 2026-04-13T14:12:58.008948+00:00 short_cpu_oracle prefill_max_delta=4.710938 prefill_cache_max_delta=4.156250 decode_max_delta=5.871094 decode_input_hidden_max_delta=0.000000 decode_step_cache_max_delta=4.156250 device_prefill_ms=1750.00 device_decode_ms=959.67

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
