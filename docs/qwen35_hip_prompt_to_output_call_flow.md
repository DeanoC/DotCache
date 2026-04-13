# Qwen35 HIP Prompt-To-Output Call Flow

This note captures the current high-level call order for the `qwen35-runtime` / `paged-runtime`
HIP path from prompt input to generated text output.

It is intentionally operational rather than exhaustive:
- which public entrypoints run first
- how prefill and decode differ
- where the model layer hands work to the HIP backend
- where HIP transport now stays off Candle on the hot path

## Scope

This describes the current greedy generation flow used by:
- [hf_qwen35_minimal.rs](/home/deano/DotCache/rust/paged-runtime/examples/hf_qwen35_minimal.rs)
- [candle_model.rs](/home/deano/DotCache/rust/paged-runtime/src/candle_model.rs)
- [model.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/model.rs)
- [hip_transport.rs](/home/deano/DotCache/rust/qwen35-runtime/src/backends/hip_transport.rs)

It is not a full page-cache planner document. It focuses on the dense runtime path and the HIP
execution boundary.

## Top-Level Flow

Pseudo-code:

```text
main()
  parse args
  load tokenizer
  load model weights
  tokenize prompt

  logits = model.forward_next_logits(prompt_token_ids)
  next_token = argmax(logits)
  generated_ids.push(next_token)

  repeat until max_new_tokens
    logits = model.forward_next_logits([next_token])
    next_token = argmax(logits)
    generated_ids.push(next_token)

  text = tokenizer.decode(prompt_ids + generated_ids)
  print text
```

Concrete entrypoints:
- example front door:
  [hf_qwen35_minimal.rs](/home/deano/DotCache/rust/paged-runtime/examples/hf_qwen35_minimal.rs)
- runtime model API:
  [candle_model.rs](/home/deano/DotCache/rust/paged-runtime/src/candle_model.rs)
- model trait surface:
  [model.rs](/home/deano/DotCache/rust/paged-runtime/src/model.rs)

## Prefill vs Decode

The runtime splits into two regimes:

1. Prefill
- input is the full prompt token sequence
- sequence length is `prompt_len`
- attention runs over the whole prompt
- linear-attention layers may choose prefill-specific fused HIP kernels

2. Decode
- input is usually one token
- sequence length is effectively `1` for the new step
- cached KV / recurrent state is reused
- linear-attention layers may choose decode-specific HIP kernels

Pseudo-code:

```text
forward_next_logits(input_ids):
  embeddings = embed(input_ids)
  hidden = embeddings

  for layer in layers:
    hidden = layer.forward(hidden, cache, mode = prefill_or_decode(input_ids))

  hidden = final_norm(hidden)
  logits = output_projection(hidden[last_position])
  return logits
```

## Model-Layer Order

At the `qwen35-runtime` model layer, the important call chain is:

- [mod.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/mod.rs)
- [model.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/model.rs)

Pseudo-code:

```text
ModelForCausalLM::forward_next_logits(input_ids)
  -> TextModel::forward_next_logits(...)
    -> token embedding lookup
    -> decoder layers in order
      -> input norm
      -> attention block
      -> post-attention norm / mlp / gated delta net
      -> residual updates
    -> final norm
    -> output projection
    -> logits
```

For a typical decoder layer:

```text
DecoderLayer::forward(...)
  residual = hidden

  hidden = attention_norm(hidden)
  attn_out = attention.forward(hidden, cache, mode)
  hidden = residual + attn_out

  residual = hidden
  hidden = mlp_or_linear_attention_norm(hidden)
  ff_out = mlp_or_gated_deltanet.forward(hidden, cache, mode)
  hidden = residual + ff_out

  return hidden
```

## Attention Path

There are two broad branches inside the current Qwen35 stack:

1. Full attention
- standard Q/K/V projection
- rotary embedding
- prefill or decode attention kernel
- output projection

2. Linear / delta attention
- fused linear projections
- conv/state update path
- recurrent scan / chunk / packed scan kernels
- readout back into token-space activations

Pseudo-code:

```text
FullAttention::forward(...)
  qkv = linear projections
  q, k, v = split
  q, k = rope(q, k)

  if prefill:
    out = backend.full_attention_prefill(q, k, v, maybe_mask)
  else:
    out = backend.full_attention_decode(q, k, v, cache)

  return out_proj(out)
```

```text
GatedDeltaNet::forward(...)
  if decode_fast_path:
    fused = backend.linear_decode_step(...)
    core_attn_out, recurrent_state = backend.unpack_linear_decode_output(fused)
  else if prefill_fast_path:
    fused = backend.linear_prefill_conv(...)
    mixed_qkv, g, conv_state = backend.unpack_linear_prefill_output(fused)
    ...

  scan/readout/update state
  return projected_output
```

## HIP Backend Boundary

The practical HIP boundary today is:

- model/backend API:
  [backend_buffer_api.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/backend_buffer_api.rs)
- backend op routing:
  [backend_ops.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/backend_ops.rs)
- HIP transport:
  [hip_transport.rs](/home/deano/DotCache/rust/qwen35-runtime/src/backends/hip_transport.rs)
- raw HIP bridge/helpers:
  [hip.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/hip.rs)

Pseudo-code:

```text
model layer
  -> backend API trait method
    -> backend_ops dispatch
      -> hip_transport fast path
        -> raw HIP kernel launch OR host/device floor composition
      -> fallback only if HIP lowering is unavailable
```

## Current HIP Transport Model

Today the hot HIP path usually tries to stay inside these storage forms:

- `OwnedDeviceBuffer`
  real HIP-owned device memory
- `MappedHostBuffer`
  host bytes mapped for HIP access
- `HostBuffer`
  plain host-backed transport storage
- `PendingHostUpload`
  lazy host result not uploaded yet

Fallback-only storage is increasingly:
- `CandleTensor`

Pseudo-code:

```text
Hip op call
  if direct owned-device kernel path exists:
    launch into OwnedDeviceBuffer
  else if mapped-host or host-buffer path exists:
    compute there
  else if native graph can stay lazy:
    return HipNativeExpr scaffold
  else:
    fall back to Candle tensor materialization
```

## Native Graph vs Materialization

The transport keeps many operations lazy through `HipNativeExpr`.

That means:
- view ops can remain lazy
- shape ops can remain lazy
- some scalar/broadcast/reduce/matmul chains can remain lazy
- actual device lowering happens later when:
  - a wrapper kernel needs materialized device inputs
  - `materialize()` is called
  - host extraction is required

Pseudo-code:

```text
HipStorage op
  if direct device floor op is available:
    return DeviceBuffer result
  else:
    return HipNativeExpr node

later:
  if host graph:
    materialize on host
  else if recursively lowerable to device:
    materialize to HipDeviceBuffer
  else:
    materialize through Candle tensor fallback
```

## Prompt-To-Output Order With HIP Focus

This is the end-to-end order in the current dense HIP path:

```text
prompt text
  -> tokenizer.encode
  -> prompt token ids

prompt token ids
  -> embedding_lookup
  -> decoder layer 0..N-1
       -> norm
       -> full attention or gated delta attention
            -> projection kernels
            -> rope / prep helpers
            -> prefill or decode kernel
            -> unpack / state update helpers
       -> residual add
       -> norm
       -> mlp / swiglu / value-decay / recurrent scan family
       -> residual add
  -> final norm
  -> output_projection
  -> logits
  -> argmax / sample on CPU side
  -> next token id

next token id
  -> decode loop repeats

all token ids
  -> tokenizer.decode
  -> output text
```

## Where Candle Still Sits

The current HIP path is no longer dominated by Candle on the hot path, but Candle still exists in:

- fallback storage:
  `HipDeviceStorage::CandleTensor`
- non-lowered fallback imports
- some cold model-side `CustomOp` scaffolding
- rare materialization tails where no owned-device or host path exists

That means the practical execution path is much cleaner than the structural dependency graph.

## Reading Order For Code

If you want to trace the runtime in code, use this order:

1. [hf_qwen35_minimal.rs](/home/deano/DotCache/rust/paged-runtime/examples/hf_qwen35_minimal.rs)
2. [candle_model.rs](/home/deano/DotCache/rust/paged-runtime/src/candle_model.rs)
3. [mod.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/mod.rs)
4. [model.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/model.rs)
5. [backend_buffer_api.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/backend_buffer_api.rs)
6. [backend_ops.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/backend_ops.rs)
7. [hip_transport.rs](/home/deano/DotCache/rust/qwen35-runtime/src/backends/hip_transport.rs)
8. [hip.rs](/home/deano/DotCache/rust/qwen35-runtime/src/qwen35_minimal/hip.rs)

That order matches the real execution stack better than reading kernels first.
