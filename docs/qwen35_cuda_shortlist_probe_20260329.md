# Qwen3.5 CUDA Shortlist Probe

This note records the first useful CUDA read on the shortlist work after pulling the latest shortlist changes onto `main`.

Scope:

- model: `Qwen/Qwen3.5-0.8B`
- backend: `torch_cuda`
- device: local `RTX 5090`
- lane: Qwen3.5 attention-subset DotCache serving
- profile: [qwen35_0p8b_attention_subset_cuda_third_pass.yaml](/workspace/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml)
- decode length: `4` generated tokens

## Short Read

The shortlist idea is useful on CUDA too.

The gain does not come from activating a better decode backend. The useful result is that shortlist reduces attended pages hard while staying on the same `per_kv_fallback` path, and that alone gives a large speedup through `16384`.

## Configs

Base shortlist:

- `execution_recent_window=1024`
- `execution_sink_window=256`
- `execution_relevance_top_k=4`
- `execution_relevance_mode=envelope`

Context-aware layer-23 variant:

- same base shortlist
- plus `execution_relevance_top_k_context_overrides=("layer:23:min_ctx:8192=8",)`

## Serving Results

Measured exact baseline on latest `main`:

| Context | Exact ms/step | Exact tok/s |
|---:|---:|---:|
| `4096` | `416.82` | `2.40` |
| `8192` | `759.04` | `1.32` |
| `16384` | `1496.27` | `0.67` |

Measured base shortlist on latest `main`:

| Context | Base shortlist ms/step | Base shortlist tok/s | Selected Pages | Candidate Pages | Speedup vs exact |
|---:|---:|---:|---:|---:|---:|
| `4096` | `205.96` | `4.86` | `4080` | `12336` | `2.02x` |
| `8192` | `203.75` | `4.91` | `4080` | `24624` | `3.73x` |
| `16384` | `251.53` | `3.98` | `4080` | `49200` | `5.95x` |

All of those shortlist runs stayed on the same decode path shape:

- `decode_path_counts={"grouped_batched": 0, "per_kv_fallback": 24}`

So the current CUDA gain is coming from page reduction, not from switching into a different backend path.

## Layer 23 Override

The first clean direct CUDA spot-check for the context-aware layer-`23` override was at `16384`:

| Context | Config | ms/step | tok/s | Selected Pages | Candidate Pages |
|---:|---|---:|---:|---:|---:|
| `16384` | base shortlist | `272.60` | `3.67` | `4080` | `49200` |
| `16384` | layer-23 `min_ctx:8192=8` | `271.63` | `3.68` | `4112` | `49200` |

That direct spot-check says the layer-`23` expansion is at least plausible on CUDA:

- it did not force a worse decode path
- it only increased selected pages slightly
- it was effectively tied on throughput at `16384`

## Quality Spot-Check

The first CUDA quality spot-check at `16384` also looked reasonable for shortlist experimentation.

Base shortlist:

- `teacher_forced_logit_max_abs_error=2.5254`
- `teacher_forced_logit_mean_abs_error=0.3031`
- `teacher_forced_logit_rmse=0.3818`
- `replay_output_max_abs_error=0.1198`

Layer-`23` context-aware override:

- slightly better logit metrics than the base shortlist in the first direct `16384` probe
- same `replay_output_max_abs_error=0.1198`

This is enough evidence to keep exploring shortlist on CUDA. It is not enough to declare a final default policy yet.

## Caveat

The `32768` CUDA automation path is not trustworthy yet.

What went wrong:

- the existing ad hoc wrappers around `bench_qwen35_attention_subset_dotcache_serving.py` left orphaned long-context processes behind
- those stale wrappers later re-launched children and polluted later attempts
- the bad state looked like very high reserved VRAM with near-zero GPU utilization and no trustworthy final JSON row

So this note intentionally stops at `16384` for the clean comparison table. `32768` still needs a dedicated single-shot runner before it should be treated as benchmark-quality evidence.

## Current CUDA Read

- shortlist helps on CUDA
- the gain is already meaningful through `16384`
- the gain does not depend on grouped batching activating
- the mechanism is simply page reduction on the existing fallback path
- the layer-`23` context-aware override looks worth keeping in the experiment set
- `32768` needs a cleaner runner before it belongs in a table

## Commands

Use the dedicated single-shot runner for future CUDA shortlist probes:

```bash
PATH=/workspace/DotCache/.venv/bin:$PATH \
PYTHONPATH=/workspace/DotCache \
.venv/bin/python scripts/run_qwen35_cuda_shortlist_probe.py \
  --contexts 4096 8192 16384 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --profile-backend \
  --output benchmarks/results/qwen35_cuda_shortlist_probe.jsonl
```

The runner does three important things the ad hoc shell wrappers did not:

- forces `--repeat-counts` empty so only exact-length probes run
- launches each benchmark in its own process group
- kills the whole process group on timeout so long-context children do not leak into later probes

The old direct benchmark commands are still useful for manual inspection, but the runner should be treated as the benchmark-quality path from here.

Exact baseline benchmark command:

```bash
PATH=/workspace/DotCache/.venv/bin:$PATH \
PYTHONPATH=/workspace/DotCache \
.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache_serving.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --target-prompt-lengths 4096 8192 16384 \
  --max-new-tokens 4 \
  --continue-on-error \
  --profile-backend
```

Base shortlist benchmark command:

```bash
PATH=/workspace/DotCache/.venv/bin:$PATH \
PYTHONPATH=/workspace/DotCache \
.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache_serving.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --target-prompt-lengths 4096 8192 16384 \
  --max-new-tokens 4 \
  --continue-on-error \
  --profile-backend \
  --execution-recent-window 1024 \
  --execution-sink-window 256 \
  --execution-relevance-top-k 4 \
  --execution-relevance-mode envelope
```

Layer-23 context-aware variant benchmark command:

```bash
PATH=/workspace/DotCache/.venv/bin:$PATH \
PYTHONPATH=/workspace/DotCache \
.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache_serving.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --target-prompt-lengths 16384 \
  --max-new-tokens 4 \
  --continue-on-error \
  --profile-backend \
  --execution-recent-window 1024 \
  --execution-sink-window 256 \
  --execution-relevance-top-k 4 \
  --execution-relevance-mode envelope \
  --execution-relevance-top-k-context-layer layer:23:min_ctx:8192=8
```
