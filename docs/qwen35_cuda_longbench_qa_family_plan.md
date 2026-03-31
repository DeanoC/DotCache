# Qwen3.5 CUDA LongBench QA Family Plan

This note defines the first non-synthetic named task family on the Qwen3.5 shortlist lane. The goal is to move beyond synthetic retrieval packs without waiting for a full LongBench reproduction.

## Chosen Family

- LongBench-derived QA mini-pack

Why this family:

- it uses real benchmark rows rather than handcrafted prompts
- it reuses the official task prompt templates from the LongBench repository
- it reuses the official LongBench QA F1 metric for the selected QA-style datasets
- it is much cheaper to run than a broad LongBench sweep

## Branch Components

- benchmark harness: [bench_qwen35_attention_subset_dotcache_longbench_qa.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_longbench_qa.py)
- prompt pack: [qwen35_cuda_longbench_qa_pack_v1.json](/Users/deanocalver/Documents/Projects/DotCache/configs/prompt_packs/qwen35_cuda_longbench_qa_pack_v1.json)
- runner: [run_qwen35_cuda_longbench_qa_probe.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_longbench_qa_probe.py)
- protocol wrapper: [run_qwen35_cuda_longbench_qa_pack_protocol.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_longbench_qa_pack_protocol.sh)
- summary tool: [summarize_qwen35_cuda_longbench_qa_pack.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/summarize_qwen35_cuda_longbench_qa_pack.py)

## Pack Shape

- `hotpotqa`, row `0`
- `2wikimqa`, row `0`
- `multifieldqa_en`, row `1`
- `qasper`, row `1`

These are fixed held-out rows from the original LongBench `data.zip` bundle, not from the earlier synthetic prompt packs.

## CUDA Command

```bash
scripts/run_qwen35_cuda_longbench_qa_pack_protocol.sh
```

Default outputs:

- [qwen35_cuda_longbench_qa_pack_protocol_v1.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_longbench_qa_pack_protocol_v1.jsonl)
- [qwen35_cuda_longbench_qa_pack_protocol_v1_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_longbench_qa_pack_protocol_v1_summary.md)

## Honest Caveats

- this is a LongBench-derived mini-pack, not a full LongBench table
- it currently covers only English QA-style tasks with the official QA F1 metric
- it is meant to fill the benchmark-breadth gap with real benchmark rows, not to replace a fuller benchmark sweep later
