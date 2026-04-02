## Qwen3.5 Matched-Budget Serving Sweep

Branch and commit:

- `codex/qwen35-9b-value-escape-scan`
- `8180247e`

Commands run on the CUDA pod:

```bash
source scripts/env_cuda.sh

.venv/bin/python scripts/run_qwen35_serving_sweep.py \
  --contexts 1024 2048 \
  --max-new-tokens 4 \
  --output-dir benchmarks/results/qwen35_serving_sweep_20260402_matched_budget \
  --skip-turboquant \
  --dotcache-profile-backend

.venv/bin/python benchmarks/bench_qwen35_attention_subset_dotcache_serving.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error \
  --profile-backend \
  --execution-recent-window 1024 \
  --execution-sink-window 256 \
  --execution-relevance-top-k 4 \
  --execution-relevance-mode envelope \
  > benchmarks/results/qwen35_serving_sweep_20260402_matched_budget/qwen35_0p8b_dotcache_shortlist_base_serving.jsonl
```

Captured outputs:

- `qwen35_0p8b_dense_serving_sweep.jsonl`
- `qwen35_0p8b_dotcache_serving_sweep.jsonl`
- `qwen35_0p8b_statecache_serving_sweep.jsonl`
- `qwen35_0p8b_dotcache_shortlist_base_serving.jsonl`

Key rows:

- dense `1024`: `dense_decode_ms_per_step=49.95`
- dense `2048`: `dense_decode_ms_per_step=35.96`
- statecache `1024`: `deltanet_statecache_decode_ms_per_step=50.65`
- statecache `2048`: `deltanet_statecache_decode_ms_per_step=13.42`
- exact DotCache `1024`: `dotcache_decode_ms_per_step=140.95`, `resident_bytes=11234304`
- exact DotCache `2048`: `dotcache_decode_ms_per_step=148.16`, `resident_bytes=22251520`
- shortlist-base DotCache `1024`: `dotcache_decode_ms_per_step=141.15`, `execution_shortlist_selected_pages=3120/3120`
- shortlist-base DotCache `2048`: `dotcache_decode_ms_per_step=111.56`, `execution_shortlist_selected_pages=4080/6192`

Notes:

- Dense, StateCache, and DotCache all produced the same generated token ids at both prompt lengths.
- The wrapper's built-in DotCache lane is the exact serving path, so the direct shortlist-base rerun is the relevant budgeted DotCache artifact here.
- The Qwen shortlist run reported: `The fast path is not available because one of the required library is not installed. Falling back to torch implementation.`
- Treat these as torch-fallback serving numbers on this pod, not final fast-path TTFT or latency claims.
