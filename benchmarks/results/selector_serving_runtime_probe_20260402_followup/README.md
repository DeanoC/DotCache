## Learned Selector Runtime Probe Follow-Up

Branch and commit:

- `codex/qwen35-9b-value-escape-scan`
- `8180247e`

Command run on the CUDA pod:

```bash
source scripts/env_cuda.sh
PROMPT="$(python3 - <<'PY'
print(' '.join(['cache locality matters for long context generation.'] * 64))
PY
)"

.venv/bin/python benchmarks/bench_llama_decode.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --prompt "$PROMPT" \
  --max-new-tokens 16 \
  --learned-page-selector-path /workspace/DotCache/benchmarks/results/llama32_selector_suite_20260401/serving_selector_artifact/linear_selector_model.json \
  --learned-page-selector-prompt-family instruction \
  --learned-page-selector-prompt-variant constraints \
  > benchmarks/results/selector_serving_runtime_probe_20260402_followup/llama32_3b_learned_selector_long_prompt.jsonl
```

Captured output:

- `llama32_3b_learned_selector_long_prompt.jsonl`

Key runtime counters:

- prompt length `513`
- `tokens_per_page=256`
- `learned_page_selector_enabled=true`
- `learned_page_selector_invocations=2688`
- `learned_page_selector_fallbacks=0`
- `learned_page_selector_prediction_counts={"M3/affine/4/float16":2688}`
- `greedy_token_agreement_rate=1.0`

Notes:

- This follow-up converts the prior Llama non-test into a real serving-path activation probe by forcing multiple full pages at the default page size.
- Dense and DotCache produced identical generated token ids for all `16` decode steps.
