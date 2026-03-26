# DotCache no-CUDA bootstrap guide for Codex

This is an addendum to the main DotCache implementation guide.

It answers a narrower question:

**How do we start verifying DotCache this week, without Triton or CUDA, on an M4 Mac and a Windows laptop with Radeon 890M?**

---

## 0. Executive decision

Yes, start now.

Do **not** wait for Triton/CUDA to begin the project. Treat this week as a **spec, correctness, and memory-traffic verification sprint**.

### Recommended machine order

1. **Primary dev box: M4 Mac**
2. **Secondary smoke-test box: Windows AMD 890M laptop**
3. **Later performance backends: CUDA / Triton / HIP**

### Recommended backend order

1. `cpu_ref` for exactness and page-layout verification
2. `torch_mps` on the Mac for vectorized execution and no-full-materialization tests
3. `mlx_metal` on the Mac only if PyTorch MPS becomes the bottleneck for the hot loops
4. `windows_rocm_smoke` or `directml_smoke` on the AMD laptop for optional validation
5. later, port only the hot kernels to Triton/CUDA

The point is simple: this week should prove the **DotCache contract**, not chase final kernel ceilings.

---

## 1. What is worth verifying before CUDA exists

You can verify most of the DotCache idea without CUDA:

- page format correctness
- grouped inner-dimension packing
- header / metadata overhead
- M0 + M3 mode behavior
- recent-window escape behavior
- exact agreement with an explicit dequantize-then-attend baseline using the **same quantized page contents**
- streaming `score_page` and `mix_page` that do **not** materialize full K or V tensors
- bytes-per-token accounting
- peak-memory accounting
- numerical error as context length grows

You should **not** expect this week to settle:

- ultimate fused-kernel speedups
- register-pressure or occupancy wins
- final hardware roofline behavior
- production runtime integration quality

That is fine. DotCache is a systems proposal with a software-first contract. The pre-CUDA job is to build the wind tunnel, not the race car.

---

## 2. Why this is still faithful to DotCache

The paper explicitly frames DotCache as a **software-first prototype** with four primitives:

```python
encode_page(tensor_slice, mode_plan) -> bitstream, header
score_page(query_slice, bitstream, header) -> partial_logits
mix_page(attn_weights, bitstream, header, out_acc) -> updated_out
choose_mode(layer, head, token_age, stats) -> mode_id
```

The paper also makes three implementation choices that matter for this no-CUDA plan:

1. **Inner-dimension grouping** is a core part of the design.
2. **Keys and values are asymmetric**, so it is valid to start with only the simple modes.
3. **M0 affine low-bit + M3 escape window** is already enough to test the central execution claim.

So the MVP for this week is:

- `M0` for K
- `M0` for V
- `M3` escape pages for recent tokens / exceptions
- page-organized storage
- fused-ish streaming execution that avoids full K/V widening

Do **not** block on M1, M2, Triton, CUDA, HIP, or vLLM integration.

---

## 3. Hardware-specific plan

## 3.1 M4 Mac, use this as the main development machine

Use the Mac as the primary box for three reasons:

1. It can run a clean CPU reference and a GPU-backed PyTorch reference on the same machine.
2. PyTorch MPS is good enough for vectorized prototype work.
3. If you need real custom kernels before CUDA arrives, Apple’s MLX stack gives you a native Metal path.

### Best path on the Mac

**Phase A:** `cpu_ref` in PyTorch or NumPy

**Phase B:** `torch_mps` backend for `score_page` and `mix_page`

**Phase C, optional:** `mlx_metal` microkernels for the hot loops only

Do not start with MLX first. Start with PyTorch because debugging is simpler and the tensor API is more familiar for Codex.

### What to keep on CPU vs GPU on the Mac

Keep this on CPU first:

- page encoding
- header packing
- bit packing
- test oracles
- reference dequantization baseline

Move this to MPS next:

- query-side group slicing
- pagewise partial-logit accumulation
- softmax
- weighted output accumulation
- peak-memory and materialization checks

If MPS handles bitwise unpacking cleanly, keep unpack on GPU too.
If not, keep **page write-time** packing work on CPU and focus the GPU path on **page read-time execution**.

The goal is not ideological purity. The goal is to verify that the **decode-time attention path** does not widen to full K/V tensors.

---

## 3.2 Windows AMD 890M laptop, use as a secondary target

Use the AMD laptop as a **secondary** target, not the first one.

### Good uses for the AMD laptop this week

- CPU reference tests
- shape / dtype compatibility checks
- optional PyTorch-on-Windows smoke runs
- optional end-to-end single-batch decode sanity checks

### Do not make it the first custom-backend target

Even though AMD now has an official Ryzen AI Windows path, Windows support is still narrower than the Mac path for this particular project.

Treat the Windows AMD machine as:

- useful for smoke tests
- useful for confirming the page format is portable
- not the best first place to build the project’s first serious backend

### If the Windows ROCm path works on your machine

Use it only for:

- single-request decode tests
- FP16 end-to-end comparisons
- confirming your abstractions are backend-agnostic

Do not let it dictate the architecture.

---

## 4. Exact project scope for the no-CUDA week

## 4.1 Modes to implement

Implement only:

- `M0`: affine low-bit quantization
- `M3`: high-precision escape pages

Defer:

- `M1`: LUT mode
- `M2`: sketch / residual inner-product mode
- heavy write-path transforms

## 4.2 Grouping and layout

Use:

- **group size**: 32 first, 64 second
- **bits**: 4-bit first, 2-bit second
- **page size**: 64 tokens or another easy power of two
- **payload layout**: group-major first for both K and V

Why group-major first:

- easiest to reason about for K-side pagewise dot products
- simplest to keep the same across CPU and MPS
- good enough for an MVP

You can revisit V layout later.

## 4.3 Backends to implement

Implement these backends in this order:

```python
Backend = Literal[
    "cpu_ref",
    "torch_mps",
    "mlx_metal",        # optional
    "windows_rocm_smoke",# optional
    "directml_smoke"     # optional fallback
]
```

Only `cpu_ref` and `torch_mps` are required this week.

---

## 5. What Codex should build first

Use this as the implementation order.

### Step 1. CPU golden model

Implement:

- `DotCacheConfig`
- `EncodedPage`
- `PageHeader`
- `encode_page`
- `decode_group_ref`
- `score_page_ref`
- `mix_page_ref`
- explicit dequantize-then-attend baseline
- pagewise attention harness

This version must be boring, readable, and obviously correct.

### Step 2. Bit packing and unpacking tests

Implement exact unit tests for:

- 2-bit packing
- 4-bit packing
- partial final-byte handling
- padding of the final group
- page header serialization / deserialization

This is the gravel inside the gearbox. Shake it out early.

### Step 3. Streaming execution path on CPU

Add a streaming path that computes:

- partial logits from compressed K pages without building a full dequantized K tensor
- weighted output from compressed V pages without building a full dequantized V tensor

It is okay if this first streaming path is still written in plain PyTorch on CPU.

### Step 4. MPS backend

Port only the execution path:

- `score_page_mps`
- `mix_page_mps`

Keep the same page format and tests.

The MPS backend should share:

- the same `EncodedPage`
- the same quantization metadata format
- the same correctness harness
- the same benchmark harness

### Step 5. Optional MLX / Metal microkernels

Only do this if the MPS path is too sluggish or too awkward.

The first two MLX kernels should be:

1. **M0 K decode-score**
   - unpack nibble or 2-bit symbols
   - apply scale and bias
   - multiply by `q_g`
   - accumulate into token logits

2. **M0 V decode-mix**
   - unpack symbols
   - apply scale and bias
   - multiply by attention weight
   - accumulate into output vector

Do **not** rewrite the whole project in MLX. Keep MLX only at the hot-kernel boundary.

---

## 6. The key architectural trick for the no-CUDA prototype

Even without a custom kernel backend, structure the code so that the execution path is already shaped like DotCache.

### Bad shape

```python
K_full = decode_entire_page_to_fp16(bitstream, header)
logits += q @ K_full.T
```

### Good shape

```python
for g in groups:
    qg = query_group[g]
    codes = load_group_codes(bitstream, g)
    partial = decode_group_on_the_fly(codes, header[g])
    logits += dot(qg, partial)
```

And similarly for values:

```python
for g in groups:
    codes = load_group_codes(bitstream, g)
    vg = decode_group_on_the_fly(codes, header[g])
    out_group += attn_weights @ vg
```

Even if the first implementation still creates a temporary tensor for a **single group**, that is acceptable.

The important part is this:

- never reconstruct the whole page tensor by default
- keep the working set group-local
- make the code path easy to swap with a later Triton/CUDA/HIP kernel

---

## 7. Acceptance criteria for the week

The project is a success this week if all of the following are true.

## 7.1 Correctness

- `encode_page` round-trips correctly for M0 and M3
- `score_page_ref` matches explicit dequantize baseline on the same codes
- `mix_page_ref` matches explicit dequantize baseline on the same codes
- `torch_mps` matches `cpu_ref` within agreed tolerances
- recent-window M3 reduces worst-case error relative to all-M0

## 7.2 Architectural verification

- page headers stay compact and measurable
- the execution path can report that it did **not** materialize a full K or V page
- peak temporary allocations are tracked
- bytes per generated token are tracked
- metadata bytes vs payload bytes are tracked

## 7.3 Benchmark verification

At minimum collect:

- encode time per page
- score time per page
- mix time per page
- total decode-step time in the harness
- peak memory
- page payload bytes
- metadata bytes
- numerical error

### A successful result does not require a speedup yet

For this week, a passing result can be:

- numerically correct
- structurally faithful to DotCache
- measurably lower peak widened memory than explicit full-page dequantization

If you also get speedups on MPS, great.
If you do not, that is still useful.

---

## 8. What to tell Codex not to do

Be explicit. Tell Codex to avoid these traps:

- do not start inside vLLM
- do not implement every mode from the paper
- do not build a fancy scheduler before the page format works
- do not optimize Python loops before the vectorized reference exists
- do not let Windows ROCm quirks reshape the core abstractions
- do not materialize full K/V pages inside the default execution path
- do not chase end-to-end model integration before page-level tests pass

---

## 9. Suggested repo layout

```text
dotcache/
  config.py
  page_format.py
  packing.py
  modes/
    m0_affine.py
    m3_escape.py
  backends/
    cpu_ref.py
    torch_mps.py
    mlx_metal.py        # optional
    windows_rocm.py     # optional smoke path
    directml_smoke.py   # optional fallback
  attention/
    baseline.py
    streaming.py
  bench/
    bench_pages.py
    bench_decode_step.py
  tests/
    test_packing.py
    test_headers.py
    test_m0_roundtrip.py
    test_m3_escape.py
    test_score_equivalence.py
    test_mix_equivalence.py
    test_no_full_materialization.py
```

---

## 10. Copy/paste brief for Codex

```text
Implement a no-CUDA, no-Triton DotCache verification prototype first.

Target machines:
- primary: Apple Silicon Mac (M4)
- secondary: Windows AMD laptop with Radeon 890M

Required backends for this stage:
- cpu_ref
- torch_mps

Optional only if needed:
- mlx_metal
- windows_rocm_smoke
- directml_smoke

Core objective:
Verify the DotCache execution contract before any CUDA/Triton work exists.
The prototype must prove that page-organized low-bit KV can be consumed through streaming score and mix paths without reconstructing full K/V tensors by default.

Implement only these modes first:
- M0 affine low-bit quantization
- M3 high-precision escape pages

Implement these primitives:
- encode_page(tensor_slice, mode_plan) -> EncodedPage
- score_page(query_slice, bitstream, header) -> partial_logits
- mix_page(attn_weights, bitstream, header, out_acc) -> updated_out
- choose_mode(layer, head, token_age, stats) -> mode_id

Constraints:
- inner-dimension grouping only
- group sizes 32 then 64
- bits 4 then 2
- page size 64 tokens first
- same page format across backends
- do not start with vLLM integration
- do not implement M1 or M2 yet
- do not materialize full-page K/V tensors in the default execution path

Required tests:
- pack/unpack exactness
- page header round-trip
- score equivalence to explicit dequant baseline using the same quantized codes
- mix equivalence to explicit dequant baseline using the same quantized codes
- recent-window M3 behavior
- no-full-materialization assertion / instrumentation

Required benchmarks:
- page encode latency
- score_page latency
- mix_page latency
- decode-step latency in a standalone attention harness
- payload bytes
- metadata bytes
- peak memory
- numerical error

Use PyTorch eager first. Prefer readability and instrumentation over cleverness.
Port only the hot execution path to MPS. If MPS becomes too awkward, add a very small MLX/Metal kernel layer only for the M0 score and mix microkernels.
```

---

## 11. Practical recommendation in one sentence

For the next week, use the **M4 Mac as the main DotCache lab**, build a **CPU + MPS streaming reference**, and treat the **AMD 890M laptop as a secondary smoke-test target**, not the project’s first custom backend.
