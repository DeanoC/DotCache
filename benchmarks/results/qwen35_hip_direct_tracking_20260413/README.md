# Qwen35 HIP Direct Tracking 2026-04-13

Status: `hip-direct` is now runnable on `hip:0` and emits JSON checkpoints.

Short checkpoint:
- generated_text: `Hello from DotCache!`
- prompt_token_count: 4
- generated_token_count: 1
- device_load_ms: 3715.45
- device_prefill_ms: 1731.58
- device_decode_ms: 445.01
- prefill_max_delta: 4.710938
- decode_max_delta: 5.808594

Current notes:
- native HIP runs still degrade the CPU reference path when CPU BF16 matmul is unsupported, but the example now continues instead of aborting.
- direct package profiles were revved to invalidate stale cached manifests while the direct metadata contract was evolving.
