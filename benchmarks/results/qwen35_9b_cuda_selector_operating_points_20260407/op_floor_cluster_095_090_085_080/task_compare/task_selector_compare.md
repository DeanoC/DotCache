# Qwen Task Selector Compare

| task | prompt_length | dense_success | dense_matches_dense_output | dense_decode_ms | exact_success | exact_matches_dense_output | exact_decode_ms | quality_success | quality_matches_dense_output | quality_decode_ms | systems_success | systems_matches_dense_output | systems_decode_ms | quality_vs_dense_speedup | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 21.423 | 1.000 | 1.000 | 257.987 | 1.000 | 1.000 | 261.810 | 1.000 | 1.000 | 261.652 | 0.082 | 1.001 | 0.999 | 0.999 | 0.138 | 0.138 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 21.405 | 1.000 | 1.000 | 430.432 | 1.000 | 1.000 | 435.421 | 1.000 | 1.000 | 436.131 | 0.049 | 0.998 | 1.000 | 1.000 | 0.163 | 0.163 |
| reasoning_arithmetic | 1024 | 0.000 | 1.000 | 21.722 | 0.000 | 1.000 | 263.381 | 0.000 | 1.000 | 266.147 | 0.000 | 1.000 | 266.263 | 0.082 | 1.000 | 0.998 | 0.998 | 0.300 | 0.300 |
| reasoning_arithmetic | 2048 | 0.000 | 1.000 | 21.653 | 0.000 | 1.000 | 440.835 | 0.000 | 1.000 | 445.486 | 0.000 | 1.000 | 444.047 | 0.049 | 1.003 | 0.995 | 0.995 | 0.205 | 0.205 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 21.888 | 1.000 | 1.000 | 258.846 | 1.000 | 1.000 | 262.468 | 1.000 | 1.000 | 262.237 | 0.083 | 1.001 | 1.000 | 1.000 | 0.341 | 0.341 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 21.693 | 1.000 | 1.000 | 436.806 | 1.000 | 1.000 | 442.633 | 1.000 | 1.000 | 440.649 | 0.049 | 1.005 | 1.002 | 1.002 | 0.226 | 0.226 |

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
