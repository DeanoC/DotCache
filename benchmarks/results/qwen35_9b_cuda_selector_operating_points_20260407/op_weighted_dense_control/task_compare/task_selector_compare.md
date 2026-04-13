# Qwen Task Selector Compare

| task | prompt_length | dense_success | dense_matches_dense_output | dense_decode_ms | exact_success | exact_matches_dense_output | exact_decode_ms | quality_success | quality_matches_dense_output | quality_decode_ms | systems_success | systems_matches_dense_output | systems_decode_ms | quality_vs_dense_speedup | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 22.183 | 1.000 | 1.000 | 260.855 | 1.000 | 1.000 | 263.176 | 1.000 | 1.000 | 263.232 | 0.084 | 1.000 | 0.999 | 0.999 | 0.138 | 0.138 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 21.471 | 1.000 | 1.000 | 434.839 | 1.000 | 1.000 | 441.921 | 1.000 | 1.000 | 440.631 | 0.049 | 1.003 | 1.000 | 1.000 | 0.163 | 0.163 |
| reasoning_arithmetic | 1024 | 0.000 | 1.000 | 22.573 | 0.000 | 1.000 | 265.147 | 0.000 | 1.000 | 267.397 | 0.000 | 1.000 | 267.339 | 0.084 | 1.000 | 0.998 | 0.998 | 0.300 | 0.300 |
| reasoning_arithmetic | 2048 | 0.000 | 1.000 | 21.811 | 0.000 | 1.000 | 443.265 | 0.000 | 1.000 | 449.141 | 0.000 | 1.000 | 450.687 | 0.049 | 0.997 | 0.995 | 0.995 | 0.205 | 0.205 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 22.142 | 1.000 | 1.000 | 261.337 | 1.000 | 1.000 | 262.686 | 1.000 | 1.000 | 263.278 | 0.084 | 0.998 | 1.000 | 1.000 | 0.341 | 0.341 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 21.786 | 1.000 | 1.000 | 441.866 | 1.000 | 1.000 | 445.946 | 1.000 | 1.000 | 445.573 | 0.049 | 1.001 | 1.002 | 1.002 | 0.226 | 0.226 |

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
| reasoning_arithmetic | 1024 | dense | 0.000 | 1.000 | no | 9 | 64 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | exact | 0.000 | 1.000 | no | 9 | 19 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | quality | 0.000 | 1.000 | no | 9 | 19 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | systems | 0.000 | 1.000 | no | 9 | 19 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | dense | 0.000 | 1.000 | no | 9 | 64 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | exact | 0.000 | 1.000 | no | 9 | 64 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | quality | 0.000 | 1.000 | no | 9 | 64 | 48 | 48 | 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | systems | 0.000 | 1.000 | no | 9 | 64 | 48 | 48 | 48\nFINAL: 48 |
| retrieval_passkey | 1024 | dense | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | exact | 1.000 | 1.000 | no | 8 | 24 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | quality | 1.000 | 1.000 | no | 8 | 24 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | systems | 1.000 | 1.000 | no | 8 | 24 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | dense | 1.000 | 1.000 | no | 9 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | exact | 1.000 | 1.000 | no | 9 | 14 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | quality | 1.000 | 1.000 | no | 9 | 14 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 2048 | systems | 1.000 | 1.000 | no | 9 | 14 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
