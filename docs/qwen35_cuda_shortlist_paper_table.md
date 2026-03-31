# Qwen3.5 CUDA Shortlist Paper Table

This note extracts the cleanest committed CUDA shortlist evidence for the Qwen3.5 `0.8B` attention-subset lane and separates it from exploratory longer-context runs that are not yet paper-grade.

## Scope

- model: `Qwen/Qwen3.5-0.8B`
- backend: `torch_cuda`
- device: local `RTX 5090`
- profile: [`qwen35_0p8b_attention_subset_cuda_third_pass.yaml`](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml)
- decode horizon: `4` generated tokens for the serving table
- shortlist base config:
  - `execution_recent_window=1024`
  - `execution_sink_window=256`
  - `execution_relevance_top_k=4`
  - `execution_relevance_mode=envelope`
- layer-23 context-aware variant:
  - base shortlist plus `layer:23:min_ctx:8192=8`

## Safe-To-Cite Table

These rows come from the clean single-shot CUDA probe summarized in [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md). This is the benchmark-quality subset to cite in the paper today.

| Context | Exact ms/step | Shortlist ms/step | Speedup | Exact tok/s | Shortlist tok/s | Selected pages | Candidate pages | Decode path |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `4096` | `416.82` | `205.96` | `2.02x` | `2.40` | `4.86` | `4080` | `12336` | `per_kv_fallback` only |
| `8192` | `759.04` | `203.75` | `3.73x` | `1.32` | `4.91` | `4080` | `24624` | `per_kv_fallback` only |
| `16384` | `1496.27` | `251.53` | `5.95x` | `0.67` | `3.98` | `4080` | `49200` | `per_kv_fallback` only |

Interpretation:

- the shortlist gain is already large through `16384`
- the gain does not depend on grouped batching activating
- the mechanism is page reduction on the same fallback decode path, not a different backend path

## Quality Spot-Check To Cite

The cleanest committed quality spot-check is the `16384` CUDA read from the same note.

Base shortlist at `16384`:

- `teacher_forced_logit_max_abs_error=2.5254`
- `teacher_forced_logit_mean_abs_error=0.3031`
- `teacher_forced_logit_rmse=0.3818`
- `replay_output_max_abs_error=0.1198`

Layer-23 context-aware override at `16384`:

- slightly better logit metrics than base shortlist in the first direct probe
- same `replay_output_max_abs_error=0.1198`
- throughput essentially tied in the clean direct spot-check:
  - base shortlist: `272.60 ms/step`
  - layer-23 override: `271.63 ms/step`

## Exploratory, Not Yet Paper-Grade

There are committed `32768` artifacts, but they should not be promoted into the main paper table yet.

Useful exploratory reads:

| Context | Config | Decode ms/step | Tail max abs logit error | Source |
| ---: | --- | ---: | ---: | --- |
| `32768` | base shortlist | `1183.58` | `3.5234` | [`qwen35_cuda_shortlist_ladder_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_ladder_quality_tail.jsonl) |
| `32768` | layer-23 context-aware | `1174.86` | `3.5273` | [`qwen35_cuda_shortlist_ladder_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_ladder_quality_tail.jsonl) |
| `32768` | base shortlist | `1385.82` | n/a | [`qwen35_cuda_shortlist_32k_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32k_probe.jsonl) |
| `32768` | layer-23 context-aware | `1274.62` | n/a | [`qwen35_cuda_shortlist_32k_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32k_probe.jsonl) |

Why these stay out of the main table:

- the older `32768` automation path had wrapper/process leakage issues
- the exact baseline for the same run family is not yet documented in one clean single-shot comparison note
- the table would mix benchmark-quality and exploratory evidence

## Exact Rerun Path For The CUDA Box

Use the dedicated wrapper script:

```bash
scripts/run_qwen35_cuda_shortlist_paper_table.sh
```

Default output:

```text
benchmarks/results/qwen35_cuda_shortlist_probe.jsonl
```

Custom output:

```bash
scripts/run_qwen35_cuda_shortlist_paper_table.sh \
  benchmarks/results/qwen35_cuda_shortlist_probe_rerun_YYYYMMDD.jsonl
```

The wrapper calls the single-shot runner and regenerates the benchmark-quality `4096/8192/16384` exact-vs-shortlist table without the old leaking-process wrapper behavior.

## Raw Source Files

- [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md)
- [`qwen35_cuda_shortlist_ladder_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_ladder_quality_tail.jsonl)
- [`qwen35_cuda_shortlist_32k_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32k_probe.jsonl)
