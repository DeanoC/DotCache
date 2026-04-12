# Qwen Task Selector Compare

| task | prompt_length | dense_success | dense_matches_dense_output | dense_decode_ms | exact_success | exact_matches_dense_output | exact_decode_ms | quality_success | quality_matches_dense_output | quality_decode_ms | systems_success | systems_matches_dense_output | systems_decode_ms | quality_vs_dense_speedup | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 22.364 | 1.000 | 1.000 | 258.822 | 1.000 | 1.000 | 385.389 | 1.000 | 1.000 | 116.196 | 0.058 | 3.317 | 0.999 | 1.001 | 0.100 | 0.084 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 21.261 | 1.000 | 1.000 | 428.973 | 1.000 | 1.000 | 711.594 | 1.000 | 1.000 | 190.877 | 0.030 | 3.728 | 1.000 | 1.001 | 0.126 | 0.113 |
| reasoning_arithmetic | 1024 | 0.000 | 1.000 | 22.614 | 0.000 | 1.000 | 262.693 | 0.000 | 1.000 | 471.089 | 0.000 | 1.000 | 121.209 | 0.048 | 3.887 | 1.003 | 0.995 | 0.142 | 0.128 |
| reasoning_arithmetic | 2048 | 0.000 | 1.000 | 21.584 | 0.000 | 1.000 | 438.402 | 0.000 | 1.000 | 822.558 | 0.000 | 1.000 | 217.320 | 0.026 | 3.785 | 0.995 | 0.986 | 0.196 | 0.184 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 22.296 | 1.000 | 1.000 | 258.424 | 1.000 | 1.000 | 400.893 | 1.000 | 1.000 | 168.892 | 0.056 | 2.374 | 1.000 | 1.000 | 0.299 | 0.304 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 21.614 | 1.000 | 1.000 | 436.365 | 1.000 | 1.000 | 697.653 | 1.000 | 1.000 | 274.277 | 0.031 | 2.544 | 1.002 | 0.999 | 0.185 | 0.182 |

## Sample Outputs

| task | prompt_length | profile | success | matches_dense_output | cap_hit | generated_tokens | decode_steps | expected | generated_first_line_cleaned | generated_text_cleaned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | dense | 1.000 | 1.000 | no | 8 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | exact | 1.000 | 1.000 | no | 8 | 31 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | quality | 1.000 | 1.000 | no | 8 | 31 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | systems | 1.000 | 1.000 | no | 8 | 31 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | dense | 1.000 | 1.000 | no | 8 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | exact | 1.000 | 1.000 | no | 8 | 31 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | quality | 1.000 | 1.000 | no | 8 | 31 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | systems | 1.000 | 1.000 | no | 8 | 31 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| reasoning_arithmetic | 1024 | dense | 0.000 | 1.000 | no | 9 | 512 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | exact | 0.000 | 1.000 | no | 9 | 19 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | quality | 0.000 | 1.000 | no | 9 | 19 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | systems | 0.000 | 1.000 | no | 9 | 19 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | dense | 0.000 | 1.000 | no | 9 | 512 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | exact | 0.000 | 1.000 | no | 9 | 236 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | quality | 0.000 | 1.000 | no | 9 | 236 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | systems | 0.000 | 1.000 | no | 9 | 236 | 48 | 48 | 48\nFINAL: 48 |
| retrieval_passkey | 1024 | dense | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | exact | 1.000 | 1.000 | no | 8 | 24 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | quality | 1.000 | 1.000 | no | 8 | 24 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | systems | 1.000 | 1.000 | no | 8 | 24 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | dense | 1.000 | 1.000 | no | 9 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | exact | 1.000 | 1.000 | no | 9 | 14 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | quality | 1.000 | 1.000 | no | 9 | 14 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | systems | 1.000 | 1.000 | no | 9 | 14 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
