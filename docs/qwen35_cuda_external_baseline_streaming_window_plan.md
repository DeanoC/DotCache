# Qwen3.5 CUDA External Baseline Plan

This note defines the cheapest honest external-comparator lane we can run immediately on the Qwen3.5 CUDA Needle pack while fuller `Quest` or `H2O` integrations remain unwired.

## Chosen comparator

We use a StreamingLLM-style sink-plus-recent-window baseline:

- sink window: `256` tokens
- recent window: `1024` tokens
- no query-aware shortlist expansion

Why this lane first:

- it is a recognizable external idea rather than another DotCache-internal ablation
- it runs on the existing Qwen3.5 attention-subset serving harness with no kernel rewrite
- it gives the paper a concrete "window baseline vs query-aware shortlist" comparison on the same prompt pack

Why this is still only a first external comparator:

- it is inspired by prior work, not a paper-faithful `Quest` or `H2O` reimplementation
- it should be presented as a sink-plus-recent reference baseline, not as a definitive reproduction claim for another system

## Canonical command

```bash
scripts/run_qwen35_cuda_streaming_window_needle_pack_protocol.sh
```

Outputs:

- [qwen35_cuda_streaming_window_needle_pack_v1.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1.jsonl)
- [qwen35_cuda_streaming_window_needle_pack_v1_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1_summary.md)

## Cases in the run

- `exact`
- `streaming_sink_recent`
- `shortlist_base`
- `shortlist_l23_ctx`

## Paper use

If this run succeeds cleanly, the comparison we want in the paper is:

- exact vs streaming window vs shortlist on the same four-prompt Needle pack
- retrieval accuracy
- exact-match rate
- mean decode ms/step with variance
- mean selected pages where applicable

The intended claim is narrow:

- query-aware shortlist should be compared against a simple non-query-aware long-context baseline, not only against exact attention

## Caveat language

If we cite this lane, the manuscript should say:

- this is a StreamingLLM-style sink-plus-recent reference baseline
- it is not a full paper-faithful reproduction of `Quest` or `H2O`
