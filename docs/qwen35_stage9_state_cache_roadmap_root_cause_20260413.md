# Qwen3.5 Stage 9 `state_cache_roadmap` Root Cause

The public `state_cache_roadmap` dense divergence is not a mixed-policy selection bug.

## Cause

The attention-subset handoff path zero-lengths the native full-attention KV tensors with
`_replace_attention_subset_cache_with_placeholders(...)`.

Qwen3.5 text decode computes RoPE positions from:

- `Qwen3_5TextModel.forward()`
- `past_key_values.get_seq_length()`

After handoff, the first full-attention layer cache is physically empty, so
`past_key_values.get_seq_length()` collapsed from the prompt length to `0`.

That made the first decode step after the shared token `12` use the wrong RoPE origin.

## Proof

On the CUDA single-case repro:

- layers `0`, `1`, and `2` matched dense exactly
- the first split was at full-attention layer `3`
- layer `3` hidden input matched dense exactly
- layer `3` RoPE inputs did not:
  - `cos_max_abs = 1.9990234375`
  - `sin_max_abs = 1.0`
- layer `3` post-RoPE query then diverged:
  - `query_max_abs = 12.62890625`

Additional direct checks:

- unpatched post-handoff `get_seq_length()` on the repro cache was `0`
- monkeypatching that single value back to `1345` restored the dense second token immediately:
  - unpatched top token: `1118`
  - patched top token: `264`

## Fix

The repo now preserves a logical attention-subset sequence length on the post-handoff cache and advances it once per generated token, even though the physical full-attention placeholder tensors stay empty.

That keeps `past_key_values.get_seq_length()` aligned with the original prompt length for Qwen3.5 position-id/RoPE computation.

## Result

After the fix, the canonical CUDA repro sequence matches dense again:

- dense: `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- hybrid after fix: `[12, 264, 11782, 314, 279, 1118, 220, 16]`

This localizes the public round-2 dense boundary to cache logical-sequence-length loss during attention-subset handoff.
