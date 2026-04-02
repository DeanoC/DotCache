## Learned Selector Runtime Probe

Branch and commit:

- `codex/qwen35-9b-value-escape-scan`
- `c3d49da3`

Commands run on the CUDA pod:

```bash
./.venv/bin/python scripts/train_page_selector_artifact.py \
  --labels /workspace/DotCache/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/labels/labels.jsonl \
  --selector-dataset /workspace/DotCache/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/labels/selector_dataset.jsonl \
  --output-dir /workspace/DotCache/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact

./.venv/bin/python scripts/train_page_selector_artifact.py \
  --labels /workspace/DotCache/benchmarks/results/llama32_selector_suite_20260401/labels/labels.jsonl \
  --selector-dataset /workspace/DotCache/benchmarks/results/llama32_selector_suite_20260401/labels/selector_dataset.jsonl \
  --output-dir /workspace/DotCache/benchmarks/results/llama32_selector_suite_20260401/serving_selector_artifact

./.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache_serving.py \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 8 \
  --tokens-per-page 16 \
  --learned-page-selector-path /workspace/DotCache/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json \
  --learned-page-selector-prompt-family cache \
  --learned-page-selector-prompt-variant locality

./.venv/bin/python benchmarks/bench_llama_decode.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --prompt "Write one short sentence about cache locality." \
  --max-new-tokens 16 \
  --learned-page-selector-path /workspace/DotCache/benchmarks/results/llama32_selector_suite_20260401/serving_selector_artifact/linear_selector_model.json \
  --learned-page-selector-prompt-family instruction \
  --learned-page-selector-prompt-variant constraints
```

Captured outputs:

- `qwen35_9b_learned_selector.jsonl`
- `llama32_3b_learned_selector.jsonl`

Key runtime counters:

- Qwen prompt length `1024`: `learned_page_selector_enabled=true`, `learned_page_selector_invocations=4992`, `learned_page_selector_fallbacks=0`, `learned_page_selector_prediction_counts={"M0/affine/4":286,"M3/affine/4/float16":4706}`
- Qwen prompt length `2048`: `learned_page_selector_enabled=true`, `learned_page_selector_invocations=13184`, `learned_page_selector_fallbacks=0`, `learned_page_selector_prediction_counts={"M0/affine/4":577,"M3/affine/4/float16":12607}`
- Llama prompt length `9`: `learned_page_selector_enabled=true`, `learned_page_selector_invocations=0`, `learned_page_selector_fallbacks=0`, `learned_page_selector_prediction_counts={}`

Notes:

- The Qwen output file also contains shorter warmup records before the `1024` and `2048` target prompt-length records.
- The Qwen run reported that the optional flash-linear-attention fast path was unavailable on this pod and it fell back to the torch implementation.
