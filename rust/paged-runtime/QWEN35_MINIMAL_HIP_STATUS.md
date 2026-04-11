# Qwen3.5 Minimal HIP Status

This note records the current status of the DotCache-local minimal HIP path for Qwen3.5, the
benchmark tools we now rely on, and the optimization directions that were tried and rejected on
the current ROCm UMA machine.

## Scope

This work is intentionally local to DotCache's minimal runtime path. It does not depend on
`candle-transformers`. The target model is the real `Qwen/Qwen3.5-0.8B` checkpoint, driven
through `candle-core` and `candle-nn`, with DotCache-owned HIP kernels where the default Candle
HIP backend was not sufficient.

Current working branches at the time of this note:

- `DotCache`: `codex/qwen35-minimal-hip`
- `candle`: `codex/qwen35-minimal-hip-support`

## What Works Now

The current minimal HIP path is a real mostly-GPU runtime, not a CPU-staged placeholder.

Major working pieces on the DotCache side:

- dense full-attention HIP prefill and decode
- fused HIP linear prefill path for the `k4/s3` conv/value-decay producer
- fused next `conv_state` update inside that same producer
- exact HIP `full_scan` path for eligible `prebatched-local`
- HIP defaults for single-chunk short-prefill and small exact multi-chunk cases
- HIP local ops for:
  - embedding lookup
  - causal mask generation
  - last-dim `cumsum`
  - `l2norm`
  - RMSNorm
  - SWiGLU
  - the delta helper surface needed by the minimal path

Major working pieces on the Candle side:

- HIP `matmul`
- HIP `to_dtype`
- HIP rotary ops
- HIP `sigmoid`
- HIP `softmax_last_dim`
- HIP `affine`
- HIP `where_cond`
- narrow contiguous HIP `reduce_op`
- HIP `cmp`

## Current Stable Behavior

On the current ROCm UMA machine, the default short-path behavior is stable and reasonably bounded.
Typical release-smoke behavior for the minimal HIP runner has been:

- `device_prefill_ms` around `1.6-1.7s`
- `device_decode_ms` around `0.63-0.68s`
- `prefill_max_delta` around `0.023-0.027`
- `decode_max_delta` around `0.021-0.027`

The combined HIP linear decode path still exists only as an experiment. It reduces transfer
traffic, but on this UMA host it is slower than the split decode path, so it should remain gated
unless it is re-evaluated on a discrete ROCm system.

## Current Loader Path

The default model-loading path is now `native`.

What that means in practice:

- package creation and reuse is owned by the `dotcache-model-store` crate
- the runtime resolves a backend/family-specific local package under `~/.cache/dotcache/model-packages/`
- the package-backed path is the default runtime path
- the old direct Hugging Face path remains only as a fallback/debug path

The public minimal-loader modes are now:

- `native`
- `direct`

The runtime also accepts:

- `DOTCACHE_QWEN35_LOAD_MODE=native|direct`

On this host, `native` is the right default. Representative `0.8B` HIP load-bench numbers:

- `native`: `load_millis≈3268`, `peak_rss_kib≈3286752`
- `direct`: `load_millis≈4849`, `peak_rss_kib≈3294900`

Important constraint:

- peak RSS is still close to `direct`
- the package bytes are mmap-backed, but backend execution storage still owns a separate live copy
- true shared-weight execution on UMA is future backend work, not part of the current loader design

The default `native` load path is now also unprofiled. Package/tensor timing collection still exists,
but only through the profiled load-bench path; the normal runtime load path no longer pays that
per-tensor accounting overhead.

Current package behavior on the native path:

- if a qwen35-minimal tensor is fully replaced by a package-built prepacked form, the raw tensor is
  no longer stored in the package
- the current examples are:
  - `conv1d.weight` -> `conv1d.weight.__dotcache_depthwise_squeezed`
  - `dt_bias` -> `dt_bias.__dotcache_head_bias_reshaped`
  - `A_log` -> `A_log.__dotcache_head_exp_reshaped`
- that reduced native reuse load time materially without changing steady-state execution behavior
- the converter also now drops entire tensor trees that the minimal runtime does not consume:
  - `model.visual.*`
  - `mtp.*`
- the native package for the minimal runtime is now text-only rather than a copy of unused
  multimodal and auxiliary weights

## Confirmed Model Size Ceiling On This Host

The minimal HIP path has now been smoke-tested beyond `0.8B` on the current host.

Confirmed working:

- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`

Confirmed not fitting on the current HIP path and machine:

- `Qwen/Qwen3.5-9B`
- `lovedheart/Qwen3.5-9B-FP8`

Important detail:

- `9B` first failed when the example loaded both a CPU runner and a HIP runner in one process
- the example now has `--device-only` so larger models can be tested without the CPU control lane
- even with `--device-only`, both the dense `9B` checkpoint and the tested FP8 variant still fail
  with `hipMalloc ... out of memory`

So the current practical upper bound to document for this specific machine/runtime combination is
`Qwen/Qwen3.5-4B`.

Representative native HIP load-bench results on this host:

- `Qwen/Qwen3.5-0.8B`: `load_millis≈3233`, `peak_rss_kib≈3286988`
- `Qwen/Qwen3.5-2B`: `load_millis≈6425`, `peak_rss_kib≈4773872`
- `Qwen/Qwen3.5-4B`: `load_millis≈9713`, `peak_rss_kib≈5519760`

## Immutable Embedding Status

There is now an experimental immutable HIP path for:

- `embed_tokens.weight`
- tied `lm_head`

It is gated by:

- `DOTCACHE_QWEN35_IMMUTABLE_EMBED=1`

This path is a real loader-side RAM win on this UMA host, but it is not a general default.

Measured behavior:

- `0.8B`
  - eager: `peak_rss_kib≈3289196`, `first_prefill_millis≈2478`
  - immutable: `peak_rss_kib≈1798792`, `first_prefill_millis≈2478`
- `2B`
  - eager: `peak_rss_kib≈4778024`, `first_prefill_millis≈2187`
  - immutable: `peak_rss_kib≈1799300`, `first_prefill_millis≈3392`
- `4B`
  - eager: `peak_rss_kib≈5523852`, `first_prefill_millis≈3649`
  - immutable: `peak_rss_kib≈1952428`, `first_prefill_millis≈16804`

So the current conclusion is:

- immutable embedding/tied output is a strong RAM-reduction tool
- it is acceptable at `0.8B` on this host
- it shifts too much cost into first prefill at `2B` and `4B`
- it should remain experimental and opt-in until there is a better transport/storage path

## Long-Context Benchmark Tools

These are the committed tools to use before touching the long-context fused prefill kernel again:

- [hf_qwen35_minimal_linear_microbench.rs](/home/deano/DotCache/rust/paged-runtime/examples/hf_qwen35_minimal_linear_microbench.rs)
- [qwen35_minimal_hip_prefill_kernel_sweep.sh](/home/deano/DotCache/rust/paged-runtime/examples/qwen35_minimal_hip_prefill_kernel_sweep.sh)

The microbench exposes fixed-token layer-local measurements plus derived kernel-proxy metrics:

- `fused_prefill_unique_input_bytes`
- `fused_prefill_unique_output_bytes`
- `fused_prefill_algorithmic_bytes`
- `fused_prefill_algorithmic_flops`
- `fused_prefill_algorithmic_arithmetic_intensity`
- `fused_prefill_achieved_gbytes_per_sec`
- `fused_prefill_achieved_gflops_per_sec`

The sweep script standardizes the `512 / 1024 / 2048` token layer-0 runs. Any future long-context
kernel change should be judged against that sweep, not by short-path smoke alone.

## Current Long-Context Wall

The dominant long-context bottleneck is still the fused linear prefill kernel, specifically the
`stage_kv_append_write_millis` bucket, which effectively tracks the same work as
`stage_linear_conv_millis`.

Representative current measurements:

- `512` tokens: hot bucket around `6.7-7.0s`
- `1024` tokens: hot bucket around `17.5-17.6s`
- `2048` tokens: hot bucket around `31.7s`

The important conclusion is that the surrounding graph is no longer the main problem. The wall is
inside the fused producer kernel itself.

## What Helped

The following changes were real wins and should be considered part of the current known-good
baseline:

- removing redundant bridge-side `hipDeviceSynchronize()` from the hot path
- fusing next `conv_state` writeback into the combined linear prefill kernel
- half-specific fast math in the hot prefill kernel
- measured launch-policy tuning:
  - `256` threads was better at `1024`
  - `128` threads was better at `2048`
  - current default switches at long sequence lengths
- adding benchmark infrastructure so long-context changes are judged by fixed-token layer runs

## What Was Tried And Rejected

The following directions were tested and rejected on this ROCm UMA host because they did not beat
the committed baseline in a defensible way:

- producer/consumer "prepared full-scan" post-producer layout transform
- row-serial long-seq fused prefill path
- retuned row-cooperative long-seq fused prefill path
- vectorized packed-write split path for the producer output
- splitting `g` / value-decay away from the combined prefill producer
- several smaller instruction-level tweaks that washed out or regressed

The practical pattern was consistent:

- correctness was often fine
- extra complexity was easy to add
- but most speculative long-context kernel variants did not materially beat the current path

That means the branch is not "unfinished". It is at a measured local optimum for this hardware,
given the designs already attempted.

## Guidance For Future Work

The next person touching this path should not restart speculative kernel surgery from scratch.
Use the committed microbench and sweep first.

Recommended process:

1. Prove the new idea on the `512`-token layer-0 gate.
2. Only if it wins there, run `1024` and `2048`.
3. Only keep changes that show a clear enough gain to justify the added complexity.

Recommended focus:

- stronger structural redesigns inside the fused linear prefill kernel
- or evaluation on a non-UMA discrete ROCm machine where transfer-heavy tradeoffs may change

Avoid spending more time on:

- small launch-shape permutations without a stronger hypothesis
- output-interface reshapes that add another staging step
- decode fusion on this UMA host unless the target hardware changes

## Bottom Line

The current branch is a solid checkpoint:

- short-path behavior is stable
- long-context profiling is instrumented properly
- the real remaining wall is known
- several tempting but unhelpful kernel directions have already been ruled out

That is enough to stop local speculative tuning here and treat the current path as the reference
implementation until either a substantially different kernel design is ready, or the work moves to
different ROCm hardware.
