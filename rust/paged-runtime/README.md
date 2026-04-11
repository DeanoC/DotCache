# dotcache-paged-runtime

This is a minimal page-native Rust runtime skeleton for DotCache experiments.

For the current status of the DotCache-local Qwen3.5 minimal HIP path, including what has landed,
what the benchmark tools are, and which long-context kernel directions were already tried and
rejected, see [QWEN35_MINIMAL_HIP_STATUS.md](/home/deano/DotCache/rust/paged-runtime/QWEN35_MINIMAL_HIP_STATUS.md).

There is also a Rust/Candle minimal-control benchmark lane for Qwen3.5. It is intentionally
separate from the main paged runtime so it can be used as a direct benchmark/control path while
the main runtime keeps evolving.

It deliberately starts smaller than `llama.cpp`:

- append-only per-sequence page tables
- explicit `KvPage` objects
- a backend trait with `prepare`, `score`, and `mix`
- a CPU reference backend for correctness work
- a virtual page layer for logical-to-physical indirection
- a session layer for prefix aliasing and copy-on-write forks
- a decode planner that resolves session page spans once per step and reuses them across heads
- feature-gated Hugging Face + Candle model loading
- a Candle-backed page backend scaffold that can target CPU, Metal, or CUDA devices
- an instrumented Llama path that writes real model-emitted KV rows through the session runtime into the paged cache

The current module split is:

- `page.rs`
  page metadata plus packed fp16 key/value rows
- `cache.rs`
  append-only sequence cache and page store
- `virtual_page.rs`
  logical page tables and physical-page alias tracking
- `backend.rs`
  backend contract, CPU reference implementation, and Candle device selectors
- `decode.rs`
  per-head decode over a list of page ids
- `model.rs`
  model-family metadata plus greedy generation helpers
- `session.rs`
  request/session state, token-wise append APIs, prefix-sharing forks, and decode planning
- `hf.rs`
  Hugging Face snapshot and safetensors discovery
- `candle_model.rs`
  actual Llama/Qwen2-family model loading and greedy decode over Candle, with Llama bound to `SessionRuntime`
- `instrumented_llama.rs`
  paged Llama forward path that emits per-layer per-kv-head KV rows into the live runtime cache

For Llama-family models, `CandleCausalLm` now also exposes session controls such as
`active_session_id`, `create_session`, `fork_session`, `fork_active_session`,
`set_active_session`, and `resolve_session_physical_page_ids` so prefix-sharing can be driven from
the model-facing API instead of only through internal runtime plumbing.
It also exposes `forward_next_logits_batch` for one-token batched decode across multiple Llama
sessions in a single model pass.

Once a Rust toolchain is available locally, the intended smoke command is:

```bash
cargo test --manifest-path rust/paged-runtime/Cargo.toml
```

To type-check the Hugging Face + Candle path as well:

```bash
cargo test --manifest-path rust/paged-runtime/Cargo.toml --features candle
```

To run the live Llama parity harness against Candle's native attention path:

```bash
cargo test --manifest-path rust/paged-runtime/Cargo.toml --features candle \
  llama_paged_logits_match_native_candle_on_tiny_hf -- --ignored --nocapture
```

The example greedy text-generation entrypoint is:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_greedy -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" 2
```

All Candle-backed examples accept `--device cpu`, `--device metal`, or `--device cuda`, plus
optional ordinals such as `metal:1` or `cuda:1`.

Metal and CUDA also need the matching crate feature at build time:

- CPU: `--features candle`
- Metal: `--features candle,candle-metal`
- CUDA: `--features candle,candle-cuda`

For example:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_greedy -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" 2 --device metal
```

That example now also reports live paged-cache stats from the same run, for example:

```text
paged physical_pages=8 virtual_pages=8 tokens=32 tokens_per_page=16
helloerdingsdelete
```

For a benchmark-style run that writes both a JSON summary and a JSONL request trace:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_bench -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-llama \
  --device metal --warmup-runs 1 --max-new-tokens 2 --tokens-per-page 16
```

That writes:

- `/tmp/dotcache-llama.summary.json`
- `/tmp/dotcache-llama.trace.jsonl`

and prints compact timing/throughput stats for the run.
The summary also records `warmup_runs` and `warmup_millis`, and the measured timings exclude that
warmup phase.

The benchmark and workload entrypoints also accept `--runtime-mode dense_control|paged_control|dotcache_experimental`.
For example, to run the new dense control lane instead of the paged runtime:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_bench -- \
  qwen2 trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 "hello" /tmp/dotcache-qwen2-dense \
  --device cpu \
  --runtime-mode dense_control \
  --warmup-runs 0 \
  --max-new-tokens 1
```

Qwen3.5 now has native Rust dense and paged lanes. The current support is:

- `qwen35` + `dense_control`: supported
- `qwen35` + `paged_control`: supported
- `qwen35` + `dotcache_experimental`: not implemented yet

Paged Qwen3.5 also has an explicit experimental compressed-page serving preset:

- `--serving-preset m3-int8`

That preset is currently intended for paged serving runs only. It applies:

- `--attention-path paged`
- `--default-key-page-mode M3/affine/4/int8`
- `--default-value-page-mode M3/affine/4/int8`

and it rejects incompatible combinations such as:

- `--runtime-mode dense_control`
- `--runtime-mode torch_control`
- `--attention-path fused`
- any explicit page-mode overrides on the same command line

For a real Qwen3.5 dense-control smoke run on Metal:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_greedy -- \
  qwen35 Qwen/Qwen3.5-0.8B "hello" 1 \
  --device metal \
  --runtime-mode dense_control
```

Those summaries now record:

- `runtime_mode`
- stage timing buckets such as `stage_qkv_projection_millis`, `stage_layout_prepare_millis`, and `stage_total_millis`
- Qwen3.5 dense-only buckets such as `stage_linear_attention_millis`, `stage_full_attention_millis`, and `stage_mlp_millis`

For matched control-baseline prompt lengths, the bench and workload entrypoints can also force
exact tokenized prompt sizes:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_bench -- \
  qwen35 Qwen/Qwen3.5-0.8B "hello" /tmp/qwen35-512 \
  --device metal \
  --runtime-mode dense_control \
  --warmup-runs 1 \
  --max-new-tokens 1 \
  --prompt-token-target 512
```

and for the mixed workload:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_workload_bench -- \
  qwen35 Qwen/Qwen3.5-0.8B "hello" /tmp/qwen35-workload \
  --device metal \
  --runtime-mode dense_control \
  --warmup-runs 1 \
  --total-sessions 4 \
  --wave-size 2 \
  --decode-rounds-per-wave 1 \
  --max-new-tokens 4 \
  --shared-prompt-token-target 512
```

For the large-model CUDA paged-vs-dense matrix on `Qwen/Qwen3.5-9B` and
`Qwen/Qwen3.5-27B`, use the benchmark runner in `benchmarks/`:

```bash
python benchmarks/bench_qwen35_paged_dense_matrix.py
```

By default that runner executes:

- single-session runs at `8192` and `32768` prompt tokens
- workload runs at `8192` prompt tokens
- dense baselines via `dense_control`
- paged runs via `paged_control --attention-path fused`
- resident page budgets `32`, `128`, and `512`
- `bf16` first, with per-model fallback to `f16` only if the first run fails for dtype support

It writes one dated output directory under `benchmarks/results/` containing:

- one subdirectory per run with stdout/stderr logs
- the raw Rust `.summary.json` and `.trace.jsonl` artifacts
- `manifest.json`
- `report.json`
- `report.md`

This pass is benchmark-first only. The current default prompt-policy table in
`rust/paged-runtime/policies/default_prompt_policies.json` is still tuned around
`Qwen/Qwen3.5-0.8B` and contexts up to `16384`, and it is intentionally left
unchanged here so larger-model measurements can be compared without silently
changing runtime policy defaults.

For the focused compressed-page comparison on the current `0.8B` paged runtime,
use the page-mode compare harness:

```bash
python benchmarks/bench_qwen35_page_mode_compare.py
```

By default that runner compares:

- `exact`
- `M3/affine/4/int8`

across:

- single-session contexts `2048` and `8192`
- workload context `2048`
- resident page budgets `32` and `128`

It writes one dated directory under `benchmarks/results/` containing:

- one subdirectory per run with stdout/stderr logs
- the raw Rust `.summary.json` and `.trace.jsonl` artifacts
- `manifest.json`
- `report.json`
- `report.md`

The current benchmark readout on CUDA is:

- `M3/int8` is slightly better than `exact` on `2048` single-session at budget `32`
- `M3/int8` is slightly worse than `exact` on `2048` single-session at budget `128`
- `M3/int8` is near-tied but still slightly worse than `exact` on `8192` single-session
- `M3/int8` is clearly better than `exact` on the `2048` workload runs at budgets `32` and `128`
- in every measured case, `M3/int8` cuts spilled bytes to about half of `exact`

So `M3/int8` is currently treated as an experimental paged serving/workload mode,
not a blanket replacement for exact pages.

For the minimal-control Qwen3.5 path, use the dedicated example:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features qwen35-minimal-cuda --example hf_qwen35_minimal_bench -- \
  Qwen/Qwen3.5-0.8B "hello" /tmp/qwen35-minimal-control \
  --device cuda:0 \
  --prompt-token-target 2048 \
  --warmup-runs 0 \
  --max-new-tokens 128
```

That writes:

- `/tmp/qwen35-minimal-control.summary.json`

and reports Luce-style headline metrics in the summary:

- `prefill_tokens_per_second`
- `decode_tokens_per_second`
- `total_tokens_per_second`

plus the minimal runtime stage buckets for the measured run.

To compare that minimal-control lane directly against the main Rust/Candle dense-control runtime,
and against a Luce-style megakernel lane that forces the minimal full-attention megakernel gates,
use the joined harness:

```bash
python benchmarks/bench_qwen35_minimal_control_compare.py --contexts 2048 8192
```

That writes one dated directory under `benchmarks/results/` containing:

- `manifest.json`
- `report.json`
- `report.md`

Each group in the report includes:

- `main_dense_control`
- `minimal_control`
- `minimal_megakernel`

for the same model, device, prompt token count, and decode length.

`minimal_megakernel` sets:

- `CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL=1`
- `CANDLE_QWEN35_HIP_PERSISTENT_FULL_PREFILL=1` on HIP only

On CUDA, this lane is still guarded by the model-side `head_dim <= 128` restriction, so
`Qwen/Qwen3.5-0.8B` currently falls back rather than claiming unsupported megakernel coverage.

To compare against the actual Luce external megakernel implementation instead of the in-tree
minimal lane, point the harness at a checkout of `https://github.com/Luce-Org/luce-megakernel`:

```bash
python benchmarks/bench_qwen35_minimal_control_compare.py \
  --contexts 2048 \
  --luce-repo /path/to/luce-megakernel
```

That adds a fourth lane:

- `luce_external_megakernel`

This lane runs the real Luce Python/CUDA extension as an external control. Current constraints:

- CUDA only
- `Qwen/Qwen3.5-0.8B` only
- context length `<= 2048`

To sweep multiple policy variants and write one summary/trace pair per variant plus an index:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_bench_sweep -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-sweep \
  --device metal \
  --warmup-runs 1 \
  --max-new-tokens 2 \
  --batch-size 2 \
  --tokens-per-page 16 \
  --resident-page-budgets none,2 \
  --restore-cooldowns 8,32
```

That writes:

- `/tmp/dotcache-sweep/index.json`
- `/tmp/dotcache-sweep/<variant>.summary.json`
- `/tmp/dotcache-sweep/<variant>.trace.jsonl`

With `--batch-size > 1`, the sweep pre-fills multiple sessions with the same prompt, then drives
batched one-token decode across those sessions and ranks variants by aggregate throughput.
If you omit `--resident-byte-budgets`, the sweep now runs a short calibration pass first and
auto-derives a more aggressive byte-budget grid from the observed peak resident bytes.

For a more realistic mixed workload with a cold seed prefill, captured-prefix reuse, staggered
session arrivals, and interleaved batched decode:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_workload_bench -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-workload \
  --device metal \
  --warmup-runs 1 \
  --total-sessions 4 \
  --wave-size 2 \
  --decode-rounds-per-wave 1 \
  --max-new-tokens 3 \
  --tokens-per-page 16 \
  --resident-page-budget 2
```

That writes:

- `/tmp/dotcache-workload.summary.json`
- `/tmp/dotcache-workload.trace.jsonl`

The summary breaks out cold shared-prefix prefill time, prefix-capture time, attached-session
suffix-prefill time, batched decode time, and one final decoded text per logical session.

To make that workload heavier without changing the shared prompt, add `--stress` to repeat each
session's unique suffix payload:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_workload_bench -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-workload-stress \
  --total-sessions 4 \
  --wave-size 2 \
  --decode-rounds-per-wave 1 \
  --max-new-tokens 3 \
  --tokens-per-page 16 \
  --stress \
  --stress-suffix-repeats 2
```

The resulting summary includes `stress_mode` and `stress_suffix_repeats` fields plus much larger
per-session suffix token counts. Stress mode works with resident page budgets, which makes it a
useful way to force real spill/restore pressure during mixed prefix-reuse workloads.

To sweep more realistic workload variants across session pressure and residency policy:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_workload_sweep -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-workload-sweep \
  --device metal \
  --warmup-runs 1 \
  --total-sessions-list 4,6 \
  --wave-sizes 2,3 \
  --decode-rounds-per-wave-list 1 \
  --max-new-tokens 3 \
  --tokens-per-page 16 \
  --restore-cooldowns 8,32
```

That writes:

- `/tmp/dotcache-workload-sweep/index.json`
- `/tmp/dotcache-workload-sweep/<variant>.summary.json`
- `/tmp/dotcache-workload-sweep/<variant>.trace.jsonl`

The index ranks variants by aggregate throughput under the mixed prefix-reuse workload and includes
session-count, wave-size, decode-round, and residency-policy metadata for each variant.
If `--resident-byte-budgets` is omitted, the sweep calibrates against one representative workload
run and auto-adds tighter byte-budget variants derived from the observed peak resident bytes.

Stress-mode sweeps are also supported:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_workload_sweep -- \
  llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-workload-stress-sweep \
  --total-sessions-list 4 \
  --wave-sizes 2 \
  --decode-rounds-per-wave-list 1 \
  --max-new-tokens 3 \
  --tokens-per-page 16 \
  --stress \
  --stress-suffix-repeats 2 \
  --restore-cooldowns 8
```

Stress-mode sweeps can be combined with explicit `--resident-page-budgets` values to compare
spill-heavy policy variants under longer attached-prefix suffix prefills.

To compare existing workload or bench artifacts after a run, use the report helper:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_workload_report -- \
  /tmp/dotcache-workload-stress-sweep
```

To compare Rust control-lane summaries against the Python dense Qwen3.5 JSONL harness output:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle,candle-metal --example hf_control_report -- \
  /tmp/dotcache-qwen35-rust-bench.summary.json \
  --python-jsonl /tmp/qwen35-python-bench.jsonl \
  --out-prefix /tmp/qwen35-control
```

That writes:

- `/tmp/qwen35-control.json`
- `/tmp/qwen35-control.md`

The control report matches Rust and Python runs by prompt length and reports deltas for:

- prefill latency
- decode latency
- total latency
- total throughput
- Rust stage timing totals for the matched run

That prints a compact Markdown ranking to stdout and writes:

- `/tmp/dotcache-workload-stress-sweep/workload-report.json`
- `/tmp/dotcache-workload-stress-sweep/workload-report.md`

The report normalizes spill and restore pressure per generated token, highlights the best
throughput and latency variants, and makes it easier to compare resident-page and cooldown policy
changes without reading raw `summary.json` files by hand.

The same command also works on synthetic bench sweep outputs, for example:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_workload_report -- \
  /tmp/dotcache-llama-batch-sweep
```

To find policies that hold up across both a synthetic bench report and a workload report, use:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_policy_report -- \
  /tmp/dotcache-llama-batch-sweep/workload-report.json \
  /tmp/dotcache-llama-workload-stress-budget-sweep/workload-report.json
```

That ranks shared policy tuples such as page budget, byte budget, and restore cooldown by a
balanced score that rewards both high throughput and low churn across all supplied report files.

To adaptively tune policy knobs within a single regime, use:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_policy_tune -- \
  bench llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-llama-tune \
  --top-k 2 \
  --max-new-tokens 2 \
  --batch-size 2 \
  --tokens-per-page 16 \
  --resident-page-budgets none,2 \
  --restore-cooldowns 8
```

That runs a coarse sweep, narrows around the most promising variants, reruns a refined sweep, and
writes:

- `/tmp/dotcache-llama-tune/tune.json`
- `/tmp/dotcache-llama-tune/tune.md`

To compare tuned recommendations across tokenizer-accurate prompt-length buckets, use:

```bash
cargo run --manifest-path rust/paged-runtime/Cargo.toml --features candle --example hf_prompt_bucket_tune -- \
  all llama trl-internal-testing/tiny-random-LlamaForCausalLM "hello" /tmp/dotcache-llama-buckets \
  --token-buckets 32,128,512 \
  --top-k 1 \
  --bench-args "--max-new-tokens 2 --batch-size 2 --tokens-per-page 16 --resident-page-budgets none,2 --restore-cooldowns 8" \
  --workload-args "--total-sessions-list 4 --wave-sizes 2 --decode-rounds-per-wave-list 1 --max-new-tokens 3 --tokens-per-page 16 --stress --stress-suffix-repeats 2 --resident-page-budgets 2,1 --restore-cooldowns 8"
```

That resolves the Hugging Face tokenizer for the selected model, constructs one prompt per target
token bucket such as `0-32`, `33-128`, or `129-512`, then runs both bench and workload adaptive
tuning for each bucket before writing:

- `/tmp/dotcache-llama-buckets/bucket-0_32/bench/tune.json`
- `/tmp/dotcache-llama-buckets/bucket-0_32/workload/tune.json`
- `/tmp/dotcache-llama-buckets/bucket-0_32/policy-cross-regime.json`
- `/tmp/dotcache-llama-buckets/bucket-33_128/bench/tune.json`
- `/tmp/dotcache-llama-buckets/bucket-33_128/workload/tune.json`
- `/tmp/dotcache-llama-buckets/bucket-33_128/policy-cross-regime.json`
- `/tmp/dotcache-llama-buckets/bucket-129_512/bench/tune.json`
- `/tmp/dotcache-llama-buckets/bucket-129_512/workload/tune.json`
- `/tmp/dotcache-llama-buckets/bucket-129_512/policy-cross-regime.json`
- `/tmp/dotcache-llama-buckets/bucket-report.json`
- `/tmp/dotcache-llama-buckets/bucket-report.md`

For a single regime, use `bench` or `workload` instead of `all` and pass common tune flags directly
after the positional arguments.

The bucket report tells you which policy the runtime should prefer for each prompt-length range,
whether that recommendation stays stable across both synthetic and mixed-workload tuning, and
whether the best setting shifts as prompts grow.

The runtime now also ships a checked-in prompt policy table at
`rust/paged-runtime/policies/default_prompt_policies.json`. `CandleCausalLm` exposes:

- `prompt_token_count(text, add_special_tokens)`
- `recommended_prompt_policy_for_token_count(token_count)`
- `recommended_prompt_policy_for_text(text, add_special_tokens)`
- `apply_prompt_policy(&policy)`
- `apply_recommended_prompt_policy_for_text(text, add_special_tokens)`

That lets callers resolve tokenizer-accurate prompt buckets and apply the measured default policy
before prefill or decode without re-running the sweep tools.
