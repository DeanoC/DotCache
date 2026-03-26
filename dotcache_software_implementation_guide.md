# DotCache software implementation guide for Codex

This document turns the DotCache paper into a **software-first implementation plan** that can be executed and verified on commodity GPUs before any hardware specialization.

It is written to be practical, staged, and slightly opinionated: build the smallest thing that tests the core claim first, then widen the aperture.

---

## 0. Copy/paste brief for Codex

Use this as the initial task brief:

> Implement a software-only prototype of DotCache, a compressed-domain KV-cache execution substrate for decode-time attention.
>
> The goal is **not** to invent a new quantizer. The goal is to verify the DotCache claim that low-bit KV should be treated as an **execution format**, not only a storage format.
>
> Build the prototype in stages:
>
> 1. Start with a standalone attention harness in PyTorch plus Triton or CUDA.
> 2. Implement a page-organized KV store with grouped low-bit encoding along the **inner dimension**.
> 3. Implement `encode_page`, `score_page`, `mix_page`, and `choose_mode`.
> 4. Support **M0 affine low-bit** and **M3 high-precision escape** first.
> 5. Keep **M1 LUT** and **M2 sketch-like key mode** behind clean interfaces, but do not block the MVP on them.
> 6. Implement compressed-domain kernels that **do not materialize full FP16/BF16 K or V tensors** by default.
> 7. Verify correctness against an explicit dequantize-then-attend baseline using the **same quantized page contents**.
> 8. Benchmark memory footprint, bytes per generated token, kernel latency, and end-to-end decode throughput versus:
>    - dense BF16/FP16 KV
>    - quantized storage + explicit dequantization + standard attention
> 9. Only after the standalone harness works, add an integration path for an existing paged runtime such as vLLM.
>
> Deliverables:
>
> - page format
> - encoder / packer
> - fused key scoring kernel
> - fused value mixing kernel
> - mode planner
> - test suite
> - microbenchmarks
> - end-to-end decode benchmark
> - short report with correctness and performance results
>
> Non-goals for MVP:
>
> - training
> - multi-node or multi-GPU
> - every codec from the paper
> - custom silicon assumptions
> - full vLLM backend implementation before standalone verification

---

## 1. What you are implementing

### Core idea

DotCache treats compressed KV pages as an **execution format**. The software path should:

- **write** K/V into compressed pages
- **read** compressed K pages directly for score computation
- **read** compressed V pages directly for weighted accumulation
- avoid reconstructing wide BF16/FP16 K/V tensors unless forced onto an escape path

### The four runtime primitives

Implement these exactly as the software contract:

```python
encode_page(tensor_slice, mode_plan) -> EncodedPage
score_page(query_slice, bitstream, header) -> partial_logits
mix_page(attn_weights, bitstream, header, out_acc) -> updated_out
choose_mode(layer, head, token_age, stats) -> mode_id
```

### What to verify

You are verifying two things separately:

1. **Numerical correctness of execution**
   - compressed-domain `score_page` and `mix_page` should match an explicit dequantize baseline using the **same** quantized codes and metadata

2. **Systems value**
   - the compressed-domain kernels should reduce widened traffic and improve the relevant latency or bandwidth metrics at sufficiently long contexts

---

## 2. Recommended scope for the MVP

### Implement first

- **K mode M0**: grouped affine low-bit quantization
- **V mode M0**: grouped affine low-bit quantization
- **Escape mode M3**: BF16 or FP16 page
- **Recent-token high-precision window**
- **Inner-dimension grouping**
- **Standalone decode-time attention harness**
- **Streaming page traversal**
- **Group sizes**: 32 and 64
- **Bit-widths**: 4-bit first, 2-bit second

### Defer until after the MVP works

- M1 LUT mode
- M2 sketch / inner-product estimator mode
- random rotation / heavy preconditioning on the write path
- vLLM-native production integration
- multi-GPU sharding
- page eviction / offload policies

### Why this scope is correct

This is enough to test the DotCache claim. M0 + M3 already captures:

- compressed page storage
- metadata handling
- grouped low-bit execution
- no-full-widening K path
- no-full-widening V path
- recent-window stabilization
- page-wise runtime scheduling

---

## 3. Recommended software stack

### Phase 1 stack

Use this first:

- **Python 3.11+**
- **PyTorch**
- **Triton** for the first kernels
- optional **CUDA C++ extension** if Triton becomes the bottleneck or makes bit-unpack awkward
- **single NVIDIA GPU**
- **decoder-only model** for end-to-end validation

### Suggested implementation sequence

1. Pure Python reference encoder and reference decode
2. PyTorch reference attention with explicit dequantization
3. Triton or CUDA compressed-domain score kernel
4. Triton or CUDA compressed-domain mix kernel
5. Standalone full attention op
6. Model integration
7. vLLM integration, if desired

### Why not start inside vLLM

Do not make your first bug hunt inside a large runtime.  
Build a small lab before rolling the boulder uphill.

The second-stage integration target should still be a paged runtime, because that aligns with DotCache. But the **first** target should be a controlled harness where every tensor is easy to inspect.

---

## 4. Architecture to implement

## 4.1 Page abstraction

A DotCache page is keyed by:

```text
(layer_id, kv_head_id, token_start:token_end, kind)
kind ∈ {K, V}
```

Use **KV head**, not query head, as the storage axis.  
For GQA or MQA models, query heads map onto KV heads at runtime.

### Page configuration

Use a config object like:

```python
@dataclass
class DotCacheConfig:
    head_dim: int
    group_size: int              # 32 or 64
    bits_k: int                  # 4 for MVP
    bits_v: int                  # 4 for MVP
    tokens_per_page: int         # e.g. 64 or runtime-aligned
    recent_window: int           # e.g. 128
    sink_window: int = 0         # optional later
    store_scales_dtype: torch.dtype = torch.float16
    store_bias_dtype: torch.dtype = torch.float16
    payload_layout_k: str = "group_major"
    payload_layout_v: str = "group_major"   # can later try token_major
    default_mode_k: str = "M0"
    default_mode_v: str = "M0"
    escape_dtype: torch.dtype = torch.bfloat16
```

---

## 4.2 Grouping rule

Partition each K or V vector into contiguous groups along the **inner dimension**:

```text
[token, head_dim] -> [token, num_groups, group_size]
num_groups = ceil(head_dim / group_size)
```

Pad the final group if `head_dim % group_size != 0`.

This is essential. DotCache and InnerQ both lean on inner-dimension grouping because it better matches the vector-matrix execution pattern.

---

## 4.3 Modes

### M0: affine low-bit

Store, per token-group:

- packed low-bit codes
- scale
- optional bias

Use this as the first working mode for both K and V.

### M3: high-precision escape

Store raw BF16 or FP16 vectors for:

- recent-token window
- outlier pages
- any token-group the planner marks as unsafe

### M1: LUT mode, later

Design the interface now, implement later.

Per group:

- tiny codebook
- packed codes

### M2: sketch / key-only inner-product mode, later

Design the interface now, implement later.

Use only on **K** pages, not V pages, in the first advanced version.

---

## 4.4 Page header and metadata layout

Do **not** make the kernel parse a fancy variable-length object graph.

Use two layers:

### Host-side rich descriptor

Useful for debugging and serialization.

```python
@dataclass
class EncodedPage:
    layer_id: int
    kv_head_id: int
    kind: Literal["K", "V"]
    token_start: int
    token_count: int
    head_dim: int
    group_size: int
    bits: int
    mode_default: int
    has_group_modes: bool
    layout: Literal["group_major", "token_major"]
    payload: torch.Tensor        # uint8 or uint32 buffer on device
    scales: torch.Tensor         # [token_count, num_groups] or packed variant
    bias: torch.Tensor | None
    group_modes: torch.Tensor | None
    exception_mask: torch.Tensor | None
    escape_payload: torch.Tensor | None
```

### Device-side compact metadata

Keep device metadata fixed-stride and pointer-light:

```python
@dataclass
class PageDescDevice:
    payload_offset: int
    scales_offset: int
    bias_offset: int
    modes_offset: int
    exception_offset: int
    token_count: int
    token_start: int
    num_groups: int
    group_size: int
    bits: int
    mode_default: int
    kind: int
    layout: int
```

### Important rule

Prefer **fixed-stride side arrays** over deep pointer chains.  
The kernel should see a neat little lunchbox, not a haunted filing cabinet.

---

## 5. Data layout details

## 5.1 K payload layout

For **K**, use `group_major` first:

```text
[group 0][token 0 packed words][token 1 packed words]...[token T-1 packed words]
[group 1][token 0 packed words][token 1 packed words]...[token T-1 packed words]
...
```

Why this is a good first choice:

- query group `q_g` is reused across all tokens
- scale or bias metadata for a token-group can be loaded with predictable strides
- the score kernel can iterate group by group and keep `q_g` hot

## 5.2 V payload layout

Start with the same layout for simplicity, but benchmark **token_major** later:

```text
[token 0][group 0 packed words][group 1 packed words]...
[token 1][group 0 packed words][group 1 packed words]...
```

Why this may help later:

- `alpha_j` is reused across all groups of token `j`
- value accumulation is not the same problem as key scoring
- DotCache explicitly allows K/V asymmetry, so their ideal physical layouts may differ

---

## 5.3 Bit packing

Use integer word packing.

Suggested representation:

- store packed symbols in `uint32`
- bit-width = 4 or 2
- packed words per group:
  - 4-bit: `group_size * 4 / 32`
  - 2-bit: `group_size * 2 / 32`

Examples:

- group size 32, 4-bit → 4 `uint32` words
- group size 32, 2-bit → 2 `uint32` words
- group size 64, 4-bit → 8 `uint32` words

Implement:

```python
pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor
unpack_bits(words: torch.Tensor, bits: int, group_size: int) -> torch.Tensor
```

Write exhaustive tests for all supported bit-widths and signed/unsigned code conventions.

---

## 6. Write path

The write path happens when a model produces fresh K and V for new tokens.

## 6.1 Write-path pipeline

Implement:

```text
K/V from model
-> optional preconditioning
-> inner-dimension grouping
-> mode selection
-> quantize or escape
-> pack low-bit symbols
-> append page payload + metadata
-> register page in page table
```

## 6.2 Optional preconditioning

For MVP, support only:

- `NONE`
- optional **key channel normalization** folded into query later

Do **not** make the MVP depend on random rotations or expensive transforms.

## 6.3 Reference M0 quantization

Implement both symmetric and affine variants behind one interface.

### Symmetric grouped quantization

```python
qmax = 2**(bits - 1) - 1
scale = max(abs(x_group)) / max(qmax, 1)
codes = clamp(round(x_group / scale), -qmax, qmax)
```

Store codes in a packed form. You may bias them into an unsigned range for storage.

### Affine grouped quantization

```python
qmin, qmax = 0, 2**bits - 1
x_min = min(x_group)
x_max = max(x_group)
scale = max((x_max - x_min) / max(qmax - qmin, 1), eps)
codes = clamp(round((x_group - x_min) / scale), qmin, qmax)
bias = x_min
```

### Recommendation

For the first end-to-end implementation:

- allow **symmetric** and **affine**
- use **affine** as default for easier accuracy
- benchmark **symmetric** too, because it simplifies kernels and metadata

## 6.4 Escape mode M3

Any group or token page can be stored in BF16/FP16 when:

- `token_age < recent_window`
- quantization error exceeds threshold
- NaN or Inf would be created
- outlier ratio exceeds threshold

This is the pressure-release valve. Without it, low-bit kernels can turn into a haunted carnival of brittle edge cases.

---

## 7. Read path for keys: `score_page`

This is the most important kernel in the prototype.

## 7.1 Functional contract

Input:

- `query_slice`: `[head_dim]` or `[num_query_heads, head_dim]`
- page descriptor for a **K** page

Output:

- partial logits for the tokens in that page:
  - `[token_count]` for one query head
  - or `[num_query_heads, token_count]` for batched heads if you choose

## 7.2 Do not dequantize into a full dense K tensor

The kernel should:

1. load page metadata
2. iterate over groups
3. unpack symbols into narrow lanes
4. compute the contribution to the dot product directly
5. accumulate logits in FP32
6. write only logits

Do **not** create a dense `[token_count, head_dim]` BF16 tensor as an intermediate.

## 7.3 Critical M0 optimization

For affine grouped quantization with one shared scale and optional bias per token-group:

```text
K̂[j, g] = scale[j, g] * code[j, g] + bias[j, g]
```

The dot product contribution is:

```text
q_g^T K̂[j, g]
= scale[j, g] * (q_g^T code[j, g]) + bias[j, g] * sum(q_g)
```

This is a key implementation trick.

### What it buys you

You do **not** need to reconstruct the decoded floating-point vector for the group.  
You only need:

- the packed codes
- `scale[j, g]`
- optionally `bias[j, g]`
- `sum(q_g)` precomputed once per group

### Implementation rule

Before iterating over tokens, precompute for the query:

```python
q_group_sums[g] = q_g.sum()
```

Then the kernel computes:

```python
int_dot = dot(q_g, code_vec)
logit_j += scale_jg * int_dot
if bias exists:
    logit_j += bias_jg * q_group_sums[g]
```

For symmetric mode:

```python
logit_j += scale_jg * dot(q_g, code_vec_signed)
```

This is one of the cleanest ways to make DotCache real in software.

## 7.4 Kernel shape for the MVP

Use one kernel launch per:

- layer
- query head or KV head tile
- page batch

A good first kernel shape is:

- one program / thread block handles one `(query_head, page)`
- inside it:
  - keep logits for that page in registers or shared memory
  - loop over groups
  - unpack codes for each token in the page
  - accumulate logits

Start simple before you get clever.

## 7.5 Escape handling in `score_page`

If the page or token-group is M3:

- use the dense BF16/FP16 data for that page/group
- still avoid reconstructing unrelated groups

For MVP, page-level escape is simpler than group-level escape.  
Implement page-level escape first.

---

## 8. Softmax strategy

You need a practical strategy between `score_page` and `mix_page`.

## 8.1 MVP softmax plan: store logits

First implementation:

1. run `score_page` over all K pages
2. write logits per token to a temporary FP32 buffer
3. compute softmax over the full token axis
4. run `mix_page` over V pages

This is not maximally fused, but it is easy to debug and still verifies the main DotCache idea: K/V are consumed in compressed form.

## 8.2 Later optimization: online softmax

After correctness is stable, add an online softmax path:

- maintain running max `m`
- maintain running normalizer `l`
- optionally avoid storing full logits

This is an optimization phase, not the first hill to die on.

---

## 9. Read path for values: `mix_page`

## 9.1 Functional contract

Input:

- attention weights for the page tokens: `[token_count]`
- page descriptor for a **V** page
- output accumulator `out_acc`: `[head_dim]`

Output:

- updated `out_acc`

## 9.2 Do not dequantize into a dense V matrix

The kernel should:

1. load page metadata
2. read `alpha_j` for tokens in the page
3. unpack codes
4. accumulate directly into the output vector
5. keep the output accumulator as the first wide representation on the path

## 9.3 M0 optimization for values

For affine grouped quantization:

```text
V̂[j, g] = scale[j, g] * code[j, g] + bias[j, g]
```

Then:

```text
out_g += alpha_j * V̂[j, g]
       = alpha_j * scale[j, g] * code[j, g] + alpha_j * bias[j, g]
```

If `bias[j, g]` is shared across the whole group, you can:

- accumulate the code contribution per lane
- accumulate a scalar `beta_g += alpha_j * bias[j, g]`
- after the token loop, add `beta_g` to every lane in the group

This avoids reconstructing the decoded vector.

## 9.4 Kernel shape for the MVP

A good first implementation:

- one program / block handles one `(query_head, page)`
- accumulate into a local output tile
- loop over tokens in the page
- unpack each token-group
- scale and add to the accumulator

Accumulate in FP32. Cast to BF16/FP16 only at the end.

## 9.5 Later asymmetry experiments

After the MVP:

- test a cheaper V metadata scheme than K
- test different layouts for V
- test whether V tolerates lower precision or lower metadata than K

This is directly aligned with the DotCache thesis that K and V are not the same animal.

---

## 10. `choose_mode`: the online planner

The planner should exist from day one, but it can be simple.

## 10.1 MVP planner

Implement a heuristic planner:

```python
def choose_mode(layer, head, token_age, stats):
    if token_age < recent_window:
        return M3
    if stats.nan_or_inf:
        return M3
    if stats.max_abs / max(stats.rms, eps) > outlier_ratio_thresh:
        return M3
    return M0
```

## 10.2 Stats to track

Per token-group or per page, collect:

- max abs value
- mean
- variance or RMS
- min / max
- fraction of outlier channels
- quantization error proxy after trial quantization
- token age

## 10.3 Later planner upgrades

Later, planner inputs can include:

- layer ID
- head ID
- token age bucket
- calibration history
- recent runtime error signals
- mode-specific error estimates

But do not start with a tiny policy network.  
A threshold planner is good enough for verification.

---

## 11. Reference implementation structure

Use this repository layout:

```text
dotcache/
  pyproject.toml
  README.md
  configs/
    dotcache_mvp.yaml
  dotcache/
    __init__.py
    config.py
    types.py
    page_format.py
    page_store.py
    planner.py
    precondition.py
    encode.py
    decode_reference.py
    attention_reference.py
    attention_runtime.py
    modes/
      __init__.py
      m0_affine.py
      m1_lut.py
      m2_sketch.py
      m3_escape.py
    kernels/
      __init__.py
      triton_pack.py
      triton_unpack.py
      triton_score_m0.py
      triton_mix_m0.py
      cuda_score_m0.cu
      cuda_mix_m0.cu
    adapters/
      hf_attention.py
      offline_trace_harness.py
      vllm_adapter/
        README.md
        notes.md
  tests/
    test_pack_unpack.py
    test_m0_quant.py
    test_page_format.py
    test_score_reference.py
    test_mix_reference.py
    test_attention_vs_dense.py
    test_escape_mode.py
    test_recent_window.py
  benchmarks/
    bench_pack.py
    bench_score.py
    bench_mix.py
    bench_decode.py
    bench_memory.py
  scripts/
    run_unit_tests.sh
    run_microbench.sh
    run_decode_bench.sh
    export_results.py
```

---

## 12. Implementation phases

## Phase 0: dense baseline and trace harness

### Goal

Create a clean baseline harness before any compression work.

### Tasks

- implement a decode-time attention benchmark over cached K/V tensors
- support:
  - one head
  - many heads
  - GQA mapping
  - multiple context lengths
- measure:
  - logits correctness
  - output correctness
  - kernel or op time
  - memory use

### Deliverable

A baseline script that can:

- run dense attention
- cache K/V
- benchmark decode one token against a long context

---

## Phase 1: pure Python reference DotCache

### Goal

Implement page packing, quantization, and exact reference decode in Python or PyTorch.

### Tasks

- implement page format
- implement M0 quantization
- implement M3 escape mode
- implement bit pack/unpack
- implement reference `score_page`
- implement reference `mix_page`
- compare against explicit dequantization

### Deliverable

A CPU or PyTorch reference that is obviously correct and painfully slow.  
That is fine. It is your oracle.

---

## Phase 2: compressed-domain K kernel

### Goal

Get `score_page` working on GPU without reconstructing dense K.

### Tasks

- Triton or CUDA kernel for M0 K pages
- page-level M3 escape fallback
- logits output in FP32
- unit test against reference implementation
- benchmark vs explicit dequantize-then-dot

### Acceptance

For identical quantized pages, compressed-domain `score_page` should match explicit dequantize scoring within numerical tolerance.

---

## Phase 3: compressed-domain V kernel

### Goal

Get `mix_page` working on GPU without reconstructing dense V.

### Tasks

- Triton or CUDA kernel for M0 V pages
- M3 fallback
- FP32 output accumulation
- unit test against reference implementation
- benchmark vs explicit dequantize-then-mix

### Acceptance

For identical quantized pages and weights, compressed-domain `mix_page` should match explicit dequantize mixing within numerical tolerance.

---

## Phase 4: end-to-end attention op

### Goal

Wire `score_page` + softmax + `mix_page` into a full decode-time attention path.

### Tasks

- page traversal across the token history
- temporary logits buffer
- softmax
- V page traversal
- GQA mapping
- causal handling
- batch=1 first, then batch>1

### Acceptance

For a fixed quantized KV cache, full DotCache attention should match explicit dequantized attention with the same cache contents.

---

## Phase 5: model integration

### Goal

Run an actual decoder-only model with DotCache-managed KV in the decode phase.

### Recommended approach

Start with one of these:

- a small Llama-family model in a Hugging Face harness
- an offline layer trace replay harness
- then a patched decode path for the full model

### Implementation guidance

- leave prefill mostly dense if needed
- encode K/V when appending to the cache
- replace only the decode-time attention path first
- keep queries in BF16 or FP16

### Acceptance

Generate text without crashes, NaNs, or shape bugs.  
Then measure accuracy drift and throughput.

---

## Phase 6: vLLM integration, optional but useful

Do this only after the standalone harness is stable.

### Why vLLM is a natural second target

Its paged attention design already traverses paged KV blocks, and its current kernels and docs explicitly expose separate `k_cache` and `v_cache` layouts. vLLM also already documents quantized KV-cache support and notes that one backend can execute attention in the quantized FP8 domain, which makes it a natural software substrate for a DotCache-style prototype.

### Suggested integration approach

- preserve the page table abstraction
- align `tokens_per_page` with the runtime block size
- implement a custom DotCache attention backend or patched kernel path
- do not fork the entire runtime unless necessary

### Practical note

The first vLLM milestone can simply be:

- use vLLM-like page tables
- swap only the attention kernel path
- keep the rest of the runtime unchanged

---

## 13. Key mathematical checks

These checks catch a shocking number of bugs.

## 13.1 Score-path equivalence

For each page and group, verify:

```text
dense_dot(q_g, decode(code, scale, bias))
==
compressed_domain_formula(q_g, code, scale, bias)
```

within tolerance.

## 13.2 Mix-path equivalence

For each page and group, verify:

```text
sum_j alpha_j * decode(code_j, scale_j, bias_j)
==
compressed_domain_mix(alpha, code, scale, bias)
```

within tolerance.

## 13.3 Pack/unpack invariants

For every supported bit-width and group size:

```text
unpack(pack(codes)) == codes
```

exactly.

## 13.4 Escape routing invariants

If a token-group is routed to M3, it must never be accidentally read through the low-bit path.

---

## 14. Correctness test plan

## 14.1 Unit tests

Write these first:

- pack / unpack roundtrip
- scale / bias quantization formulas
- page descriptor serialization
- page layout indexing
- M3 escape roundtrip
- planner threshold behavior

## 14.2 Property tests

Randomized tests over:

- bit-width
- group size
- head dimension
- token count
- value ranges
- symmetric vs affine mode
- escape fraction

Use random seeds and keep failing examples.

## 14.3 Kernel parity tests

For random pages:

- reference `score_page` vs GPU `score_page`
- reference `mix_page` vs GPU `mix_page`

## 14.4 Full attention parity

For a fixed quantized cache:

- explicit dequantize attention vs DotCache attention

This is the load-bearing test.  
It isolates execution correctness from quantization quality.

## 14.5 End-to-end model checks

Then measure:

- next-token agreement rate versus dense baseline
- perplexity on a held-out corpus
- qualitative generation sanity
- stability over long contexts

---

## 15. Benchmark plan

## 15.1 Baselines to implement

At minimum benchmark against:

1. **Dense BF16 or FP16 KV**
2. **Quantized storage + explicit dequantize-then-attend**
3. **DotCache compressed-domain execution**

If practical, later compare against:

- existing KV quantization paths
- vLLM FP8 KV cache paths
- alternative runtime designs

## 15.2 Benchmark grid

Run over:

- context lengths: 1k, 4k, 8k, 16k, 32k, 64k
- batch sizes: 1, 4, 8, 16 where memory allows
- group size: 32 and 64
- bit-width: 4 then 2
- recent-window sizes: 0, 64, 128, 256

## 15.3 Metrics to collect

Collect all of these:

- peak KV memory footprint
- payload bytes per token
- metadata bytes per token
- temporary logits bytes
- encode time per appended token
- `score_page` latency
- `mix_page` latency
- end-to-end decode latency
- tokens per second
- achieved bandwidth if you can collect it
- fraction of tokens or groups routed to M3

## 15.4 Most important comparison

For the same quantized page contents:

```text
explicit dequantization path
vs
compressed-domain path
```

This directly tests the DotCache claim.

---

## 16. Performance instrumentation

Implement a small metrics layer.

### Log these counters

- total page payload bytes read
- total metadata bytes read
- total dense escape bytes read
- total logits bytes written
- total output bytes written
- number of page fetches
- number of escape pages
- number of escape groups, if group-level escape exists

### Use these timers

- Python wall-clock for end-to-end
- CUDA events for kernels
- optional profiler hooks for Nsight or PyTorch profiler

### Derived metrics

Compute:

```text
bytes_per_generated_token
metadata_fraction
escape_fraction
score_time_fraction
mix_time_fraction
```

Without these, you are flying by starlight with the dashboard unplugged.

---

## 17. Acceptance criteria

Use these as the MVP bar.

## 17.1 Functional acceptance

- `encode_page`, `score_page`, `mix_page`, `choose_mode` all implemented
- M0 and M3 fully working
- standalone full attention path working
- no dense K/V reconstruction on the compressed-domain kernels
- GQA supported
- recent-window escape supported

## 17.2 Numerical acceptance

For the same quantized cache contents:

- compressed-domain score path matches explicit dequant score within tolerance
- compressed-domain mix path matches explicit dequant mix within tolerance
- full attention output matches explicit dequant attention within tolerance

Use FP32 accumulation so tolerances stay tight.

## 17.3 Systems acceptance

Produce a report that includes:

- memory footprint reduction
- metadata overhead
- crossover point where DotCache beats explicit dequantization, if any
- cases where DotCache loses and why

Do not hide the losing cases. Those are often the map.

---

## 18. Common failure modes

Expect these. They arrive early and with enthusiasm.

### Quantization and packing bugs

- off-by-one in signed code ranges
- wrong zero offset for stored signed codes
- broken padding on last group
- endian mistakes in packing
- wrong packed-word stride

### Metadata bugs

- mixing token-major and group-major assumptions
- wrong page offsets after appends
- stale page descriptors after reallocation
- wrong scale or bias indexing

### Numerical bugs

- accumulating in BF16 instead of FP32
- forgetting the `bias * sum(q_g)` term
- mismatched softmax mask behavior
- bad escape-path routing

### Runtime bugs

- incorrect KV-head mapping under GQA
- querying pages in the wrong token order
- recent-window pages stored compressed instead of escaped
- race conditions in output accumulation

---

## 19. Extensions after the MVP

## 19.1 M1 LUT mode

Implement a small per-group codebook.

Useful experiments:

- compare LUT metadata cost versus affine metadata cost
- compare K-only, V-only, and K+V LUT deployment

## 19.2 M2 key-only sketch mode

Implement only after M0 is stable.

Ideas:

- per-group random projection sketches
- approximate inner-product estimator
- optional residual correction path

Benchmark this only on the K path.

## 19.3 Better K/V asymmetry

Try:

- heavier metadata for K, lighter metadata for V
- group-major K, token-major V
- 4-bit K and 2-bit V
- different recent-window policies for K and V

## 19.4 Optional preconditioning

After the MVP, add:

- per-channel key normalization folded into query
- Hadamard-like transforms if justified
- write-path-only transforms that do not burden the score-critical read path

---

## 20. What to tell Codex not to do

Be explicit.

- Do **not** start with all four modes.
- Do **not** start inside a complex runtime.
- Do **not** optimize before you have a reference oracle.
- Do **not** silently widen K or V into dense intermediates in the compressed-domain kernels.
- Do **not** benchmark against a different quantization scheme when verifying execution correctness.
- Do **not** conflate quantization loss with execution bugs.
- Do **not** make page metadata variable-length inside the first GPU kernel.
- Do **not** ship without counters for metadata overhead.

---

## 21. Final recommended milestone plan

### Milestone A
Dense baseline + offline trace harness

### Milestone B
Python reference page format + M0 + M3

### Milestone C
GPU `score_page` for K

### Milestone D
GPU `mix_page` for V

### Milestone E
Full decode-time attention

### Milestone F
End-to-end model integration

### Milestone G
Benchmark report

### Milestone H
Optional vLLM integration

---

## 22. Minimal report template

Ask Codex to produce this when done.

```md
# DotCache MVP results

## Configuration
- model:
- GPU:
- group size:
- bits_k:
- bits_v:
- tokens per page:
- recent window:
- layouts:
- symmetric vs affine:

## Correctness
- pack/unpack:
- score parity:
- mix parity:
- full attention parity:
- end-to-end generation sanity:

## Memory
- dense KV footprint:
- quantized storage footprint:
- DotCache payload bytes:
- DotCache metadata bytes:
- escape bytes:
- net compression ratio:

## Performance
- dense decode latency:
- explicit dequant decode latency:
- DotCache decode latency:
- score kernel time:
- mix kernel time:
- crossover context length:

## Observations
- where DotCache wins:
- where DotCache loses:
- likely causes:
- next experiments:
```

---

## 23. Sources and design anchors

This guide is grounded in:

- the DotCache draft itself, especially the page-organized compressed-domain design, the asymmetric K/V execution model, the software contract, and the evaluation plan
- InnerQ’s emphasis on **inner-dimension grouping**, high-precision windows, and hardware-aware decode execution
- vLLM’s paged attention design and existing quantized KV-cache support as a plausible second-stage integration target

If you want a single sentence to remember the whole thing:

> Build a paged low-bit KV cache that can be **executed directly**, verify it against an explicit dequantization baseline using the same codes, and only then climb into a larger serving runtime.