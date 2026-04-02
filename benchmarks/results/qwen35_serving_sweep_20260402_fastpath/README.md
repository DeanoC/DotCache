## Qwen3.5 Serving Sweep With Fast Path Active

Branch and commit:

- `codex/qwen35-9b-value-escape-scan`
- `cdad3c02`

Commands run on the CUDA pod:

```bash
source scripts/env_cuda.sh

.venv/bin/python scripts/run_qwen35_serving_sweep.py \
  --contexts 1024 2048 \
  --max-new-tokens 4 \
  --output-dir benchmarks/results/qwen35_serving_sweep_20260402_fastpath \
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
  > benchmarks/results/qwen35_serving_sweep_20260402_fastpath/qwen35_0p8b_dotcache_shortlist_base_serving.jsonl
```

Captured outputs:

- `qwen35_0p8b_dense_serving_sweep.jsonl`
- `qwen35_0p8b_dotcache_serving_sweep.jsonl`
- `qwen35_0p8b_statecache_serving_sweep.jsonl`
- `qwen35_0p8b_dotcache_shortlist_base_serving.jsonl`

Key rows:

- dense `1024`: `dense_decode_ms_per_step=19.67`
- dense `2048`: `dense_decode_ms_per_step=9.03`
- statecache `1024`: `deltanet_statecache_decode_ms_per_step=26.23`
- statecache `2048`: `deltanet_statecache_decode_ms_per_step=11.57`
- exact DotCache `1024`: `dotcache_decode_ms_per_step=111.28`, `resident_bytes=11234304`
- exact DotCache `2048`: `dotcache_decode_ms_per_step=144.83`, `resident_bytes=22244352`
- shortlist-base DotCache `1024`: `dotcache_decode_ms_per_step=114.77`, `execution_shortlist_selected_pages=3120/3120`
- shortlist-base DotCache `2048`: `dotcache_decode_ms_per_step=111.96`, `execution_shortlist_selected_pages=4080/6192`

Comparison against the earlier fallback sweep in `qwen35_serving_sweep_20260402_matched_budget`:

- dense `1024`: `49.95 -> 19.67 ms/step`
- dense `2048`: `35.96 -> 9.03 ms/step`
- statecache `1024`: `50.65 -> 26.23 ms/step`
- statecache `2048`: `13.42 -> 11.57 ms/step`
- shortlist-base DotCache `1024`: `141.15 -> 114.77 ms/step`
- shortlist-base DotCache `2048`: `111.56 -> 111.96 ms/step`

Notes:

- Dense, StateCache, and DotCache still produced matching generated token ids at both prompt lengths.
- The wrapper's built-in DotCache lane remains the exact serving path, so the direct shortlist-base rerun is still the relevant budgeted DotCache artifact.
- The earlier fast-path fallback warning did not appear on these runs.
