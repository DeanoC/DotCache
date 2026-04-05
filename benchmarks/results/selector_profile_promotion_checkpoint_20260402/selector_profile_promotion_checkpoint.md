# Selector Profile Promotion Checkpoint

## Promotion Call

- Qwen3.5 4B / 9B / 27B: promote `systems` as the default serving selector profile. The new matrix shows the same basic read across all three validated Qwen sizes.
- Llama 3.2 3B: keep `quality` and `systems` as equivalent operating points for now. The selector is already saturated to `M3`, so the extra systems tuning does not unlock additional speed.

## Cross-Family Overview

| model | task_success_exact | task_success_quality | task_success_systems | mean_systems_vs_quality_speedup | promotion_call |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 4B | 1.000 | 1.000 | 1.000 | 2.676 | Promote systems as default |
| Qwen3.5 9B | 1.000 | 1.000 | 1.000 | 3.496 | Promote systems as default |
| Qwen3.5 27B | 0.833 | 0.833 | 0.833 | 3.510 | Promote systems as default |
| Llama 3.2 3B | 1.000 | 1.000 | 1.000 | 0.981 | Quality and systems equivalent |

## Qwen Serving-Quality Check

| context | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | token_agreement_quality | token_agreement_systems | quality_rmse | systems_rmse | quality_m3_frac | systems_m3_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 67.656 | 45.229 | 1.496 | 1.000 | 1.000 | 0.470 | 0.467 | 0.954 | 0.988 |
| 2048 | 82.911 | 49.993 | 1.658 | 1.000 | 1.000 | 0.278 | 0.272 | 0.965 | 0.999 |

## Cross-Family Task Check

| model | task | prompt_length | exact_success | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_success | systems_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 4B | instruction_constraints | 1024 | 1.000 | 193.947 | 71.346 | 2.718 | 1.000 | 1.000 |
| Qwen3.5 4B | instruction_constraints | 2048 | 1.000 | 345.015 | 108.500 | 3.180 | 1.000 | 1.000 |
| Qwen3.5 4B | reasoning_arithmetic | 1024 | 1.000 | 179.060 | 74.802 | 2.394 | 1.000 | 1.000 |
| Qwen3.5 4B | reasoning_arithmetic | 2048 | 1.000 | 295.819 | 127.593 | 2.318 | 1.000 | 1.000 |
| Qwen3.5 4B | retrieval_passkey | 1024 | 1.000 | 205.690 | 79.788 | 2.578 | 1.000 | 1.000 |
| Qwen3.5 4B | retrieval_passkey | 2048 | 1.000 | 346.413 | 120.692 | 2.870 | 1.000 | 1.000 |
| Qwen3.5 9B | instruction_constraints | 1024 | 1.000 | 141.544 | 47.499 | 2.980 | 1.000 | 1.000 |
| Qwen3.5 9B | instruction_constraints | 2048 | 1.000 | 175.314 | 63.062 | 2.780 | 1.000 | 1.000 |
| Qwen3.5 9B | reasoning_arithmetic | 1024 | 1.000 | 177.417 | 45.869 | 3.868 | 1.000 | 1.000 |
| Qwen3.5 9B | reasoning_arithmetic | 2048 | 1.000 | 234.885 | 57.693 | 4.071 | 1.000 | 1.000 |
| Qwen3.5 9B | retrieval_passkey | 1024 | 1.000 | 179.586 | 48.393 | 3.711 | 1.000 | 1.000 |
| Qwen3.5 9B | retrieval_passkey | 2048 | 1.000 | 247.253 | 69.296 | 3.568 | 1.000 | 1.000 |
| Qwen3.5 27B | instruction_constraints | 1024 | 1.000 | 356.516 | 118.880 | 2.999 | 1.000 | 1.000 |
| Qwen3.5 27B | instruction_constraints | 2048 | 1.000 | 633.198 | 150.473 | 4.208 | 1.000 | 1.000 |
| Qwen3.5 27B | reasoning_arithmetic | 1024 | 1.000 | 348.192 | 114.787 | 3.033 | 1.000 | 1.000 |
| Qwen3.5 27B | reasoning_arithmetic | 2048 | 1.000 | 585.709 | 148.847 | 3.935 | 1.000 | 1.000 |
| Qwen3.5 27B | retrieval_passkey | 1024 | 1.000 | 332.201 | 110.049 | 3.019 | 1.000 | 1.000 |
| Qwen3.5 27B | retrieval_passkey | 2048 | 0.000 | 560.837 | 145.165 | 3.863 | 0.000 | 0.000 |
| Llama 3.2 3B | instruction_constraints | 1024 | 1.000 | 59.064 | 59.036 | 1.000 | 1.000 | 1.000 |
| Llama 3.2 3B | instruction_constraints | 2048 | 1.000 | 72.796 | 76.909 | 0.947 | 1.000 | 1.000 |
| Llama 3.2 3B | reasoning_arithmetic | 1024 | 1.000 | 58.832 | 59.290 | 0.992 | 1.000 | 1.000 |
| Llama 3.2 3B | reasoning_arithmetic | 2048 | 1.000 | 72.677 | 72.207 | 1.007 | 1.000 | 1.000 |
| Llama 3.2 3B | retrieval_passkey | 1024 | 1.000 | 58.867 | 60.173 | 0.978 | 1.000 | 1.000 |
| Llama 3.2 3B | retrieval_passkey | 2048 | 1.000 | 72.817 | 75.772 | 0.961 | 1.000 | 1.000 |

## Qwen LongBench External Check

| model | context_cap | exact_qa_f1 | quality_qa_f1 | systems_qa_f1 | streaming_qa_f1 | exact_decode_ms | quality_decode_ms | systems_decode_ms | streaming_decode_ms | systems_vs_quality_speedup | systems_vs_streaming_speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 4B | 4096 | 0.253 | 0.253 | 0.253 | 0.253 | 625.448 | 1165.355 | 373.600 | 258.878 | 3.119 | 0.693 |
| Qwen3.5 4B | 8192 | 0.247 | 0.247 | 0.247 | 0.247 | 1051.394 | 2147.806 | 800.082 | 271.088 | 2.684 | 0.339 |
| Qwen3.5 9B | 4096 | 0.441 | 0.441 | 0.441 | 0.441 | 632.705 | 592.300 | 96.195 | 260.797 | 6.157 | 2.711 |
| Qwen3.5 9B | 8192 | 0.291 | 0.291 | 0.291 | 0.291 | 1046.409 | 787.975 | 152.936 | 274.512 | 5.152 | 1.795 |
| Qwen3.5 27B | 4096 | 0.358 | 0.358 | 0.358 | 0.358 | 1263.516 | 1737.076 | 331.793 | 538.435 | 5.235 | 1.623 |
| Qwen3.5 27B | 8192 | 0.341 | 0.341 | 0.341 | 0.341 | 2096.277 | 3512.227 | 804.331 | 566.767 | 4.367 | 0.705 |

## Qwen Backend Check

| model | context | exact_decode_ms | shortlist_decode_ms | learned_decode_ms | learned_vs_exact_speedup | learned_vs_shortlist_speedup | learned_m3_frac | learned_score_ms | learned_mix_ms | selector_us_per_invocation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5 4B | 1024 | 232.187 | 242.411 | 71.470 | 3.249 | 3.392 | 0.982 | 23.231 | 21.980 | 25.204 |
| Qwen3.5 4B | 2048 | 400.156 | 300.465 | 122.571 | 3.265 | 2.451 | 0.971 | 55.860 | 37.622 | 25.193 |
| Qwen3.5 9B | 1024 | 242.515 | 242.638 | 74.782 | 3.243 | 3.245 | 0.988 | 18.306 | 26.048 | 25.876 |
| Qwen3.5 9B | 2048 | 404.701 | 307.733 | 105.716 | 3.828 | 2.911 | 0.999 | 33.260 | 36.804 | 25.524 |
| Qwen3.5 27B | 1024 | 485.963 | 504.257 | 149.393 | 3.253 | 3.375 | 0.995 | 39.705 | 35.910 | 24.829 |
| Qwen3.5 27B | 2048 | 821.412 | 626.305 | 236.010 | 3.480 | 2.654 | 0.995 | 81.960 | 68.358 | 24.959 |

## Notes

- The Qwen matrix now gives a single family-wide read across `4B`, `9B`, and native `27B` on compact tasks, LongBench, and backend truth.
- All compact Qwen task rows pass except the previously-known shared `27B @ 2048 retrieval_passkey` miss, which still fails in `exact`, `quality`, and `systems` and therefore does not read like a selector regression.
- Llama task rows confirm the same task success profile, but with `systems` and `quality` effectively tied on decode.
- The Qwen LongBench matrix currently behaves more like a comparative systems check than a selector-separating quality benchmark: quality is tied across methods on the present pack, but `systems` remains much faster than `quality` and `exact` and retains lower RMSE than the streaming reference.
- Backend truth stays consistent across Qwen scale: learned lanes are strongly M3-heavy, selector overhead stays around `25 us/inv`, and the remaining dominant cost is still backend `score + mix` rather than selector computation.
- This checkpoint supports the current repo policy: Qwen serving defaults to `systems`, while Llama does not need extra systems bias.
