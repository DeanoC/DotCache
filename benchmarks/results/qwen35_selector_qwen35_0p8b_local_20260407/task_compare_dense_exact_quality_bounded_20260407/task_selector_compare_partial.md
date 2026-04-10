# Qwen Task Selector Compare (Dense / Exact / Quality, Partial)

| task | prompt_length | dense_success | dense_matches_dense_output | dense_decode_ms | exact_success | exact_matches_dense_output | exact_decode_ms | quality_success | quality_matches_dense_output | quality_decode_ms | quality_vs_dense_speedup | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 1.000 | 1.000 | 61.659 | 1.000 | 1.000 | 239.601 | 0.000 | - | 0.000 | - | - | - | - | - | - |
| instruction_constraints | 1024 | 1.000 | 1.000 | 54.381 | 1.000 | 1.000 | 267.447 | 0.000 | - | 0.000 | - | - | - | - | - | - |
| reasoning_arithmetic | 512 | 0.000 | 1.000 | 73.057 | 0.000 | 1.000 | 302.134 | 0.000 | - | 0.000 | - | - | - | - | - | - |
| retrieval_passkey | 512 | 1.000 | 1.000 | 79.007 | 1.000 | 1.000 | 236.769 | 1.000 | 1.000 | 532.517 | 0.148 | - | 1.000 | - | 0.204 | - |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 107.048 | 1.000 | 1.000 | 249.686 | 0.000 | - | 0.000 | - | - | - | - | - | - |

## Sample Outputs

| task | prompt_length | profile | success | matches_dense_output | cap_hit | generated_tokens | decode_steps | expected | generated_first_line_cleaned | generated_text_cleaned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | dense | 1.000 | 1.000 | no | 10 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 512 | exact | 1.000 | 1.000 | no | 10 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | dense | 1.000 | 1.000 | no | 10 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | exact | 1.000 | 1.000 | no | 10 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| reasoning_arithmetic | 512 | dense | 0.000 | 1.000 | yes | 512 | 512 | 48 | 17 + 26 - 9 + 14 = 44 | 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - |
| reasoning_arithmetic | 512 | exact | 0.000 | 1.000 | yes | 512 | 512 | 48 | 17 + 26 - 9 + 14 = 44 | 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - |
| retrieval_passkey | 512 | dense | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 512 | exact | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 512 | quality | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | dense | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
| retrieval_passkey | 1024 | exact | 1.000 | 1.000 | no | 8 | 64 | RIVER-58142 | RIVER-58142 | RIVER-58142 |
