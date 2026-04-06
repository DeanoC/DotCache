# Qwen3.5 CUDA Shortlist Large-Context Plan

This note defines the exact rerun path for the next large-context CUDA read on the Qwen3.5 `0.8B` attention-subset lane.

## Goal

Answer one narrow question cleanly:

> At `32768` and `49152` context tokens, does the shortlist path produce a real, stable systems win over the no-shortlist baseline, and does the layer-23 context-aware widening help?

## Important Baseline Fix

Use the `third_pass` layer profile as the shared base for these runs:

- [`qwen35_0p8b_attention_subset_cuda_third_pass.yaml`](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml)

Do **not** use [`qwen35_0p8b_attention_subset_cuda_shortlist_baseline.yaml`](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_shortlist_baseline.yaml) for the exact rows, because that profile already bakes in execution shortlist settings and a `layer:23:min_ctx:32768=8` override.

For the large-context baseline question, the clean setup is:

- exact: `third_pass` profile, no execution shortlist knobs
- shortlist base: `third_pass` profile plus shortlist knobs from the command line
- shortlist `layer-23` context-aware: same as shortlist base plus `layer:23:min_ctx:32768=8`

## Serving Command

Run this on the real CUDA box:

```bash
scripts/run_qwen35_cuda_shortlist_large_context_serving.sh
```

Equivalent raw command:

```bash
PATH=/workspace/DotCache/.venv/bin:$PATH \
PYTHONPATH=/workspace/DotCache \
.venv/bin/python scripts/run_qwen35_cuda_shortlist_probe.py \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --contexts 32768 49152 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --timeout-seconds 900 \
  --profile-backend \
  --output benchmarks/results/qwen35_cuda_shortlist_large_context_probe.jsonl
```

## Quality-Tail Command

Run this immediately after the serving probe on the same box:

```bash
scripts/run_qwen35_cuda_shortlist_large_context_quality_tail.sh
```

Equivalent raw command:

```bash
PATH=/workspace/DotCache/.venv/bin:$PATH \
PYTHONPATH=/workspace/DotCache \
.venv/bin/python scripts/run_qwen35_cuda_shortlist_probe.py \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --contexts 32768 49152 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --timeout-seconds 900 \
  --quality-check \
  --quality-mode loss_tail \
  --quality-eval-steps 4 \
  --profile-backend \
  --output benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail.jsonl
```

## What To Record

For each context and case:

- `dotcache_decode_ms_per_step`
- `decode_path_counts`
- shortlist page counts
- `teacher_forced_logit_max_abs_error`
- if available, `teacher_forced_logit_mean_abs_error`
- if available, `teacher_forced_logit_rmse`
- any timeout / OOM / missing-row failures

## Fill-In Table Template

Use this table once the two artifacts are generated:

| Context | Case | Decode ms/step | Speedup vs exact | Decode path | Selected pages | Candidate pages | Tail max abs logit error | Result |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `32768` | exact |  | `1.00x` |  | n/a | n/a |  |  |
| `32768` | shortlist base |  |  |  |  |  |  |  |
| `32768` | shortlist `layer:23:min_ctx:32768=8` |  |  |  |  |  |  |  |
| `49152` | exact |  | `1.00x` |  | n/a | n/a |  |  |
| `49152` | shortlist base |  |  |  |  |  |  |  |
| `49152` | shortlist `layer:23:min_ctx:32768=8` |  |  |  |  |  |  |  |

## Decision Rule

Promote the larger-context shortlist story only if all of the following are true:

- the runs come from the clean single-shot wrapper path
- exact rows are truly no-shortlist rows
- shortlist remains bounded as context grows
- decode path is understood and reported
- the shortlist win is stable at least once at `32768`, ideally also at `49152`
- quality does not regress badly enough to erase the systems value

If those conditions do not hold, keep the paper claim narrow and report the large-context lane as ongoing investigation rather than a locked result.
