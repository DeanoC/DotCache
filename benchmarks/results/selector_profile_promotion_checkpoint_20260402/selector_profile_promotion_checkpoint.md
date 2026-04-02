# Selector Profile Promotion Checkpoint

## Promotion Call

- Qwen3.5 9B: promote `systems` as the default serving selector profile. It preserves task success while delivering materially lower decode latency than `quality`.
- Llama 3.2 3B: keep `quality` and `systems` as equivalent operating points for now. The selector is already saturated to `M3`, so the extra systems tuning does not unlock additional speed.

## Cross-Family Overview

| model | task_success_exact | task_success_quality | task_success_systems | mean_systems_vs_quality_speedup | promotion_call |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 9B | 1.000 | 1.000 | 1.000 | 3.679 | Promote systems as default |
| Llama 3.2 3B | 1.000 | 1.000 | 1.000 | 0.981 | Quality and systems equivalent |

## Qwen Serving-Quality Check

| context | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | token_agreement_quality | token_agreement_systems | quality_rmse | systems_rmse | quality_m3_frac | systems_m3_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 67.656 | 45.229 | 1.496 | 1.000 | 1.000 | 0.470 | 0.467 | 0.954 | 0.988 |
| 2048 | 82.911 | 49.993 | 1.658 | 1.000 | 1.000 | 0.278 | 0.272 | 0.965 | 0.999 |

## Cross-Family Task Check

| model | task | prompt_length | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_success | systems_success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 9B | instruction_constraints | 1024 | 129.965 | 42.836 | 3.034 | 1.000 | 1.000 |
| Qwen3.5 9B | instruction_constraints | 2048 | 166.003 | 54.497 | 3.046 | 1.000 | 1.000 |
| Qwen3.5 9B | reasoning_arithmetic | 1024 | 172.191 | 41.983 | 4.101 | 1.000 | 1.000 |
| Qwen3.5 9B | reasoning_arithmetic | 2048 | 233.687 | 54.198 | 4.312 | 1.000 | 1.000 |
| Qwen3.5 9B | retrieval_passkey | 1024 | 166.330 | 45.144 | 3.684 | 1.000 | 1.000 |
| Qwen3.5 9B | retrieval_passkey | 2048 | 230.647 | 59.217 | 3.895 | 1.000 | 1.000 |
| Llama 3.2 3B | instruction_constraints | 1024 | 59.064 | 59.036 | 1.000 | 1.000 | 1.000 |
| Llama 3.2 3B | instruction_constraints | 2048 | 72.796 | 76.909 | 0.947 | 1.000 | 1.000 |
| Llama 3.2 3B | reasoning_arithmetic | 1024 | 58.832 | 59.290 | 0.992 | 1.000 | 1.000 |
| Llama 3.2 3B | reasoning_arithmetic | 2048 | 72.677 | 72.207 | 1.007 | 1.000 | 1.000 |
| Llama 3.2 3B | retrieval_passkey | 1024 | 58.867 | 60.173 | 0.978 | 1.000 | 1.000 |
| Llama 3.2 3B | retrieval_passkey | 2048 | 72.817 | 75.772 | 0.961 | 1.000 | 1.000 |

## Notes

- Qwen task rows come from the strengthened reasoning task slice, which now passes in `exact`, `quality`, and `systems`.
- Llama task rows confirm the same task success profile, but with `systems` and `quality` effectively tied on decode.
- This checkpoint supports the current repo policy: Qwen serving defaults to `systems`, while Llama does not need extra systems bias.
