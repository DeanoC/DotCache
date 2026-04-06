## Learned Selector Runtime Probe With Qwen Fast Path Active

Branch and commit:

- `codex/qwen35-9b-value-escape-scan`
- `cdad3c02`

Command run on the CUDA pod:

```bash
source scripts/env_cuda.sh

.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache_serving.py \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 8 \
  --tokens-per-page 16 \
  --learned-page-selector-path /workspace/DotCache/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json \
  --learned-page-selector-prompt-family cache \
  --learned-page-selector-prompt-variant locality \
  > benchmarks/results/selector_serving_runtime_probe_20260402_fastpath/qwen35_9b_learned_selector.jsonl
```

Captured output:

- `qwen35_9b_learned_selector.jsonl`

Key runtime counters:

- prompt length `1024`: `learned_page_selector_invocations=4992`, `learned_page_selector_fallbacks=0`, `learned_page_selector_prediction_counts={"M0/affine/4":287,"M3/affine/4/float16":4705}`, `dotcache_decode_ms_per_step=60.40`
- prompt length `2048`: `learned_page_selector_invocations=13184`, `learned_page_selector_fallbacks=0`, `learned_page_selector_prediction_counts={"M0/affine/4":577,"M3/affine/4/float16":12607}`, `dotcache_decode_ms_per_step=68.56`

Comparison against the earlier fallback run in `selector_serving_runtime_probe_20260402`:

- `1024`: decode `65.36 -> 60.40 ms/step`
- `2048`: decode `71.45 -> 68.56 ms/step`

Notes:

- Selector activation behavior stayed intact under the fast path: same nonzero invocations, no fallbacks, and still nontrivial `M0` versus `M3` choices at both target contexts.
- The earlier `flash-linear-attention` fallback warning did not appear on this run.
