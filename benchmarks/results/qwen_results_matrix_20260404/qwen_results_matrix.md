# Qwen Results Matrix V1

## Coverage

| model | task_rows | longbench_rows | backend_rows |
| --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 6 | 8 | 2 |
| Qwen/Qwen3.5-9B | 6 | 8 | 2 |
| Qwen/Qwen3.5-27B | 6 | 8 | 2 |

## Task Compare Matrix

| model | task | context | exact_success | quality_success | systems_success | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-27B | instruction_constraints | 1024 | 1.000 | 1.000 | 1.000 | 356.516 | 118.880 | 2.999 | 0.486 | 0.484 |
| Qwen/Qwen3.5-27B | instruction_constraints | 2048 | 1.000 | 1.000 | 1.000 | 633.198 | 150.473 | 4.208 | 0.466 | 0.468 |
| Qwen/Qwen3.5-27B | reasoning_arithmetic | 1024 | 1.000 | 1.000 | 1.000 | 348.192 | 114.787 | 3.033 | 0.401 | 0.404 |
| Qwen/Qwen3.5-27B | reasoning_arithmetic | 2048 | 1.000 | 1.000 | 1.000 | 585.709 | 148.847 | 3.935 | 0.325 | 0.336 |
| Qwen/Qwen3.5-27B | retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 332.201 | 110.049 | 3.019 | 0.682 | 0.693 |
| Qwen/Qwen3.5-27B | retrieval_passkey | 2048 | 0.000 | 0.000 | 0.000 | 560.837 | 145.165 | 3.863 | 0.330 | 0.328 |
| Qwen/Qwen3.5-4B | instruction_constraints | 1024 | 1.000 | 1.000 | 1.000 | 193.947 | 71.346 | 2.718 | 0.206 | 0.206 |
| Qwen/Qwen3.5-4B | instruction_constraints | 2048 | 1.000 | 1.000 | 1.000 | 345.015 | 108.500 | 3.180 | 0.374 | 0.361 |
| Qwen/Qwen3.5-4B | reasoning_arithmetic | 1024 | 1.000 | 1.000 | 1.000 | 179.060 | 74.802 | 2.394 | 0.278 | 0.272 |
| Qwen/Qwen3.5-4B | reasoning_arithmetic | 2048 | 1.000 | 1.000 | 1.000 | 295.819 | 127.593 | 2.318 | 0.367 | 0.362 |
| Qwen/Qwen3.5-4B | retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 205.690 | 79.788 | 2.578 | 0.545 | 0.545 |
| Qwen/Qwen3.5-4B | retrieval_passkey | 2048 | 1.000 | 1.000 | 1.000 | 346.413 | 120.692 | 2.870 | 0.598 | 0.591 |
| Qwen/Qwen3.5-9B | instruction_constraints | 1024 | 1.000 | 1.000 | 1.000 | 141.544 | 47.499 | 2.980 | 0.247 | 0.242 |
| Qwen/Qwen3.5-9B | instruction_constraints | 2048 | 1.000 | 1.000 | 1.000 | 175.314 | 63.062 | 2.780 | 0.294 | 0.288 |
| Qwen/Qwen3.5-9B | reasoning_arithmetic | 1024 | 1.000 | 1.000 | 1.000 | 177.417 | 45.869 | 3.868 | 1.052 | 1.046 |
| Qwen/Qwen3.5-9B | reasoning_arithmetic | 2048 | 1.000 | 1.000 | 1.000 | 234.885 | 57.693 | 4.071 | 1.000 | 1.004 |
| Qwen/Qwen3.5-9B | retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 179.586 | 48.393 | 3.711 | 0.463 | 0.460 |
| Qwen/Qwen3.5-9B | retrieval_passkey | 2048 | 1.000 | 1.000 | 1.000 | 247.253 | 69.296 | 3.568 | 0.592 | 0.582 |

## LongBench Matrix

| model | context_cap | case | exact_match | qa_f1 | decode_ms | decode_p95_ms | ppl_ratio | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-27B | 4096 | exact | 0.250 | 0.358 | 1263.516 | 1273.398 | - | 0.521 |
| Qwen/Qwen3.5-27B | 4096 | quality | 0.250 | 0.358 | 1737.076 | 1761.137 | - | 0.513 |
| Qwen/Qwen3.5-27B | 4096 | streaming_sink_recent | 0.250 | 0.358 | 538.435 | 542.084 | - | 0.779 |
| Qwen/Qwen3.5-27B | 4096 | systems | 0.250 | 0.358 | 331.793 | 335.305 | - | 0.511 |
| Qwen/Qwen3.5-27B | 8192 | exact | 0.250 | 0.341 | 2096.277 | 2117.435 | - | 0.564 |
| Qwen/Qwen3.5-27B | 8192 | quality | 0.250 | 0.341 | 3512.227 | 3581.215 | - | 0.550 |
| Qwen/Qwen3.5-27B | 8192 | streaming_sink_recent | 0.250 | 0.341 | 566.767 | 569.926 | - | 0.948 |
| Qwen/Qwen3.5-27B | 8192 | systems | 0.250 | 0.341 | 804.331 | 813.295 | - | 0.550 |
| Qwen/Qwen3.5-4B | 4096 | exact | 0.000 | 0.253 | 625.448 | 633.926 | - | 0.519 |
| Qwen/Qwen3.5-4B | 4096 | quality | 0.000 | 0.253 | 1165.355 | 1210.864 | - | 0.496 |
| Qwen/Qwen3.5-4B | 4096 | streaming_sink_recent | 0.000 | 0.253 | 258.878 | 270.252 | - | 1.203 |
| Qwen/Qwen3.5-4B | 4096 | systems | 0.000 | 0.253 | 373.600 | 388.383 | - | 0.491 |
| Qwen/Qwen3.5-4B | 8192 | exact | 0.000 | 0.247 | 1051.394 | 1069.017 | - | 0.519 |
| Qwen/Qwen3.5-4B | 8192 | quality | 0.000 | 0.247 | 2147.806 | 2186.643 | - | 0.532 |
| Qwen/Qwen3.5-4B | 8192 | streaming_sink_recent | 0.000 | 0.247 | 271.088 | 275.845 | - | 1.189 |
| Qwen/Qwen3.5-4B | 8192 | systems | 0.000 | 0.247 | 800.082 | 806.332 | - | 0.534 |
| Qwen/Qwen3.5-9B | 4096 | exact | 0.250 | 0.441 | 632.705 | 649.039 | - | 0.414 |
| Qwen/Qwen3.5-9B | 4096 | quality | 0.250 | 0.441 | 592.300 | 612.570 | - | 0.385 |
| Qwen/Qwen3.5-9B | 4096 | streaming_sink_recent | 0.250 | 0.441 | 260.797 | 265.246 | - | 1.046 |
| Qwen/Qwen3.5-9B | 4096 | systems | 0.250 | 0.441 | 96.195 | 97.778 | - | 0.383 |
| Qwen/Qwen3.5-9B | 8192 | exact | 0.000 | 0.291 | 1046.409 | 1053.647 | - | 0.361 |
| Qwen/Qwen3.5-9B | 8192 | quality | 0.000 | 0.291 | 787.975 | 800.225 | - | 0.333 |
| Qwen/Qwen3.5-9B | 8192 | streaming_sink_recent | 0.000 | 0.291 | 274.512 | 278.413 | - | 1.040 |
| Qwen/Qwen3.5-9B | 8192 | systems | 0.000 | 0.291 | 152.936 | 154.320 | - | 0.331 |

## Backend Truth Matrix

| model | context | exact_decode_ms | shortlist_decode_ms | learned_decode_ms | learned_vs_exact_speedup | learned_vs_shortlist_speedup | learned_m3_frac | selector_us_per_invocation | learned_score_ms | learned_mix_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3.5-27B | 1024 | 485.963 | 504.257 | 149.393 | 3.253 | 3.375 | 0.995 | 24.829 | 39.705 | 35.910 |
| Qwen/Qwen3.5-27B | 2048 | 821.412 | 626.305 | 236.010 | 3.480 | 2.654 | 0.995 | 24.959 | 81.960 | 68.358 |
| Qwen/Qwen3.5-4B | 1024 | 232.187 | 242.411 | 71.470 | 3.249 | 3.392 | 0.982 | 25.204 | 23.231 | 21.980 |
| Qwen/Qwen3.5-4B | 2048 | 400.156 | 300.465 | 122.571 | 3.265 | 2.451 | 0.971 | 25.193 | 55.860 | 37.622 |
| Qwen/Qwen3.5-9B | 1024 | 242.515 | 242.638 | 74.782 | 3.243 | 3.245 | 0.988 | 25.876 | 18.306 | 26.048 |
| Qwen/Qwen3.5-9B | 2048 | 404.701 | 307.733 | 105.716 | 3.828 | 2.911 | 0.999 | 25.524 | 33.260 | 36.804 |
