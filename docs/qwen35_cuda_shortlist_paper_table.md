# Qwen3.5 CUDA Shortlist Paper Table

This note extracts the cleanest committed CUDA shortlist evidence for the Qwen3.5 `0.8B` attention-subset lane and separates the latest rerun from older, more optimistic historical probe notes.

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

## Current Rerun To Cite

The current paper-facing artifact is the dedicated rerun in [`qwen35_cuda_shortlist_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl), summarized in the `2026-03-31` entry of [`performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md). This is the latest honest read on the CUDA lane.

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Read |
| ---: | ---: | ---: | ---: | --- |
| `4096` | `257.14` | `164.88` | `163.02` | shortlist win |
| `8192` | `167.94` | `167.37` | `171.77` | effectively flat |
| `16384` | `194.55` | `198.41` | `201.60` | shortlist slightly worse |

What this means:

- the wrapper rerun is valid and reproducible
- the shortlist lane still runs cleanly at all three contexts
- all nine rerun rows stayed on `per_kv_fallback`
- the long-context speedup claim is **not** stable enough to cite as a current result

So the current safe paper claim is narrower:

- shortlist is operational on CUDA
- shortlist is helpful at `4096`
- shortlist is not yet a stable long-context throughput win on this lane
- grouped-batched decode remains the main unresolved systems blocker

## Historical Probe Worth Mentioning, Not Leading With

The older March 29 probe note in [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md) is still useful context because it recorded a much stronger earlier result:

| Context | Exact ms/step | Base shortlist ms/step | Speedup |
| ---: | ---: | ---: | ---: |
| `4096` | `416.82` | `205.96` | `2.02x` |
| `8192` | `759.04` | `203.75` | `3.73x` |
| `16384` | `1496.27` | `251.53` | `5.95x` |

That older note is now best treated as an encouraging historical probe, not the current paper table, because the dedicated rerun did not reproduce the `8192` and `16384` wins.

## Quality Spot-Check

The cleanest committed quality spot-check is still the `16384` CUDA read from the older March 29 note.

Base shortlist at `16384`:

- `teacher_forced_logit_max_abs_error=2.5254`
- `teacher_forced_logit_mean_abs_error=0.3031`
- `teacher_forced_logit_rmse=0.3818`
- `replay_output_max_abs_error=0.1198`

Layer-23 context-aware override at `16384`:

- slightly better logit metrics than base shortlist in the first direct probe
- same `replay_output_max_abs_error=0.1198`
- throughput essentially tied in that direct spot-check:
  - base shortlist: `272.60 ms/step`
  - layer-23 override: `271.63 ms/step`

## Exploratory, Not Yet Paper-Grade

There are now committed larger-context artifacts from the clean wrapper path, but they still belong in a secondary table because the quality read is mixed.

Current large-context serving rerun from [`qwen35_cuda_shortlist_large_context_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe.jsonl):

| Context | Exact ms/step | Base shortlist ms/step | Layer-23 ctx ms/step | Decode-path read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `2298.36` | `673.12` | `671.80` | all rows `per_kv_fallback` |
| `49152` | `3675.23` | `786.35` | `844.04` | all rows `per_kv_fallback` |

Current large-context quality-tail rerun from [`qwen35_cuda_shortlist_large_context_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl):

| Context | Exact tail max abs logit error | Base shortlist | Layer-23 ctx | Read |
| ---: | ---: | ---: | ---: | --- |
| `32768` | `0.8984` | `3.5098` | `3.5137` | shortlist much noisier than exact |
| `49152` | `4.5742` | `7.0000` | `6.9648` | shortlist still materially worse; layer-23 widening does not cleanly fix it |

What this adds to the current paper read:

- the shortlist path is a real serving-speed win at `32768` and `49152`
- shortlist page counts remain bounded relative to the full no-shortlist path
- grouped-batched decode still does not activate
- the `49152` quality-tail read is not clean enough to present as a settled win
- the layer-23 context-aware widening is neutral at `32768` and slightly worse on serving speed at `49152`

Why these stay out of the main table:

- the large-context speed story is real, but the quality story is not yet clean at `49152`
- all rows still stay in `per_kv_fallback`, so the mechanism remains incomplete
- the paper should not present these larger-context rows as a fully locked systems result yet

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

The wrapper calls the single-shot runner and regenerates the current `4096/8192/16384` exact-vs-shortlist comparison without the old leaking-process wrapper behavior.

## Raw Source Files

- [`qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md)
- [`qwen35_cuda_shortlist_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl)
- [`qwen35_cuda_shortlist_large_context_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_probe.jsonl)
- [`qwen35_cuda_shortlist_large_context_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl)
- [`performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md)
- [`qwen35_cuda_shortlist_ladder_quality_tail.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_ladder_quality_tail.jsonl)
- [`qwen35_cuda_shortlist_32k_probe.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_shortlist_32k_probe.jsonl)
