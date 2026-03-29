# Qwen3.5 DotCache Decode Pseudocode

This note describes the current serving-style DotCache decode path for `Qwen/Qwen3.5-0.8B` and the measured sizes it operates on. The goal is to make the hot path easy to reason about before changing kernels or runtime structure.

## Model And Runtime Shape

- Model: `Qwen/Qwen3.5-0.8B`
- Full-attention layers targeted by DotCache: `3, 7, 11, 15, 19, 23`
- DotCache layer count: `6`
- Query heads: `8`
- KV heads: `2`
- Query heads per KV head: `4`
- Head dim: `256`
- Group size: `32`
- Groups per head: `8`
- Padded head dim: `256`
- Tokens per page: `16`

So for one decode step on one targeted layer:

- `q` shape is `[8, 256]`
- `k_step` shape is `[2, 1, 256]`
- `v_step` shape is `[2, 1, 256]`
- grouped query input to the backend is `2` KV groups, each with `4` query heads
- padded grouped query tensor is `[2, 4, 8, 32]`

## Decode Pseudocode

```text
for each decode token:
    run the native Qwen3.5 layer stack

    for each targeted full-attention layer in [3, 7, 11, 15, 19, 23]:
        q, k_step, v_step = project_current_hidden_state()
        append k_step and v_step into the per-layer per-kv-head persistent tail

        query_groups = split q into 2 groups by kv head
        prepared_key_pages, prepared_value_pages = gather static pages + live tail page

        for each kv group:
            pad the 4 query heads from [4, 256] to [4, 8, 32]
            score padded queries against all prepared key pages
            concatenate per-chunk logits into a token-length axis
            softmax over all attended tokens
            mix the weights against the prepared value pages

        context = merge the 2 kv-group outputs back into [8, 256]
        output = o_proj(context)
        continue native layer execution
```

The backend function that dominates time today is the grouped prepared decode in [torch_mps.py](/workspace/DotCache/dotcache/backends/torch_mps.py), specifically `decode_grouped_multiquery_step_prepared_torch_tensor(...)`.

## Backend Pseudocode

This is the current mental model of the hot backend path for one targeted layer:

```text
query_groups: 2 groups, each [4, 256]
key_pages_by_group: 2 page lists, one per kv head
value_pages_by_group: 2 page lists, one per kv head

prepared_query_groups = pad_queries(query_groups)  # [2, 4, 8, 32]

for each aligned page chunk:
    chunk_key_pages = slice key page lists for both kv groups
    logits_chunk = score(prepared_query_groups, chunk_key_pages)
    append logits_chunk

logits = concat(logits_chunks, token_axis)  # [2, 4, total_tokens]
weights = softmax(logits, token_axis)

output = zeros([2, 4, 256])
for each aligned page chunk:
    chunk_weights = slice weights for this chunk
    chunk_value_pages = slice value page lists for both kv groups
    output += mix(chunk_weights, chunk_value_pages)

return output reshaped back to [8, 256]
```

The current trace does not show a meaningful `prepare_ms_total` bucket. That means the page-list preparation above is either outside the timed sections or too small relative to the decode core to matter in the current instrumentation.

## Per-Context Working Set

For exact-length serving runs with backend profiling enabled:

| Context | Pages per KV Head per Layer | KV Pages per Layer | Total Key Pages Across 6 Layers | Total Value Pages Across 6 Layers | Logits per Layer | Logits Across 6 Layers |
|---:|---:|---:|---:|---:|---:|---:|
| `4096` | `256` | `512` | `3072` | `3072` | `32768` | `196608` |
| `16384` | `1024` | `2048` | `12288` | `12288` | `131072` | `786432` |
| `32768` | `2048` | `4096` | `24576` | `24576` | `262144` | `1572864` |

`Logits per layer` above comes from `2 KV groups * 4 query heads per group * context_length`.

## Measured Decode Breakdown

Profile artifact: [qwen35_0p8b_dotcache_serving_profile.jsonl](/workspace/DotCache/benchmarks/results/qwen35_dotcache_profile_20260329/qwen35_0p8b_dotcache_serving_profile.jsonl)

| Context | Step ms | Decode Runtime ms | QKV ms | Append ms | Output ms | Score ms | Mix ms | Softmax ms | Chunk Assembly ms | Unpack ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4096` | `401.33` | `380.79` | `2.24` | `0.74` | `0.38` | `200.59` | `161.14` | `0.21` | `0.40` | `0.00` |
| `16384` | `1490.67` | `1469.07` | `2.30` | `0.79` | `0.44` | `761.30` | `642.16` | `0.28` | `0.43` | `1.62` |
| `32768` | `2961.35` | `2939.84` | `2.30` | `0.80` | `0.44` | `1662.27` | `1155.47` | `0.32` | `0.45` | `11.88` |

What this says:

- `qkv`, `append`, and `output` are effectively flat.
- Long-context growth is almost entirely inside backend decode.
- Inside backend decode, `score` and `mix` dominate.
- `softmax` and `chunk_assembly` are tiny.
- `unpack` starts to matter at `32768`, but it is still much smaller than `score` or `mix`.

## Mode Mix At The Page Level

These page counts are totals across the 6 targeted layers:

| Context | Key M0 Pages | Key M2 Pages | Value M0 Pages | Value M1 Pages | Value M3 Pages |
|---:|---:|---:|---:|---:|---:|
| `4096` | `1029` | `2043` | `2044` | `4` | `1024` |
| `16384` | `4852` | `7436` | `8188` | `4` | `4096` |
| `32768` | `9990` | `14586` | `16380` | `4` | `8192` |

So the current third-pass layer profile is mostly:

- key pages in `M0` and `M2`
- value pages in `M0` and `M3`
- a negligible `M1` value slice

## What To Conclude

- The expensive thing is not append overhead and not projection overhead.
- The expensive thing is iterating the prepared page set during grouped decode, then scoring and mixing over it.
- From `16384 -> 32768`, step time grows almost exactly `2x`, which matches the page-count and token-count growth.
- DotCache is behaving like an attention-surface optimization that still pays near-linear work over the remaining attended pages.

## Instrumentation Caveats

- `prepare_ms_total` is `0` in this trace. Treat that as an instrumentation gap, not proof that there is no page-preparation work.
- `host_to_device_bytes` is `0` in the backend trace, so the current hot path does not look transfer-bound.
- `prepared_page_cache_hits` and `cache_resident_bytes` are also `0` in the backend trace even though the serving result reports non-zero prepared-page cache residency. Those cache counters are not fully surfaced on this path yet, so the timing numbers are more trustworthy than the cache counters here.
