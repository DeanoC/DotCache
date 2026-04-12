# Qwen Task Selector Compare

| task | prompt_length | dense_success | dense_matches_dense_output | dense_decode_ms | exact_success | exact_matches_dense_output | exact_decode_ms | quality_success | quality_matches_dense_output | quality_decode_ms | systems_success | systems_matches_dense_output | systems_decode_ms | quality_vs_dense_speedup | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 21.958 | 1.000 | 1.000 | 262.241 | 1.000 | 1.000 | 264.221 | 1.000 | 1.000 | 263.670 | 0.083 | 1.002 | 0.999 | 0.999 | 0.138 | 0.138 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 21.600 | 1.000 | 1.000 | 435.797 | 1.000 | 1.000 | 439.118 | 1.000 | 1.000 | 439.516 | 0.049 | 0.999 | 1.000 | 1.000 | 0.163 | 0.163 |
| reasoning_arithmetic | 1024 | 0.000 | 1.000 | 22.201 | 0.000 | 1.000 | 266.269 | 0.000 | 1.000 | 269.303 | 0.000 | 1.000 | 268.259 | 0.082 | 1.004 | 0.998 | 0.998 | 0.300 | 0.300 |
| reasoning_arithmetic | 2048 | 0.000 | 1.000 | 22.385 | 0.000 | 1.000 | 445.441 | 0.000 | 1.000 | 449.490 | 0.000 | 1.000 | 448.485 | 0.050 | 1.002 | 0.995 | 0.995 | 0.205 | 0.205 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 21.894 | 1.000 | 1.000 | 262.301 | 1.000 | 1.000 | 264.774 | 1.000 | 1.000 | 263.585 | 0.083 | 1.005 | 1.000 | 1.000 | 0.341 | 0.341 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 22.037 | 1.000 | 1.000 | 441.177 | 1.000 | 1.000 | 446.482 | 1.000 | 1.000 | 445.043 | 0.049 | 1.003 | 1.002 | 1.002 | 0.226 | 0.226 |

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
