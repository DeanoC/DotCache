# Qwen3.5 9B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 205.351 | 129.565 | 42.963 | 3.016 | 0.043 | 0.012 |
| instruction_constraints | 2048 | 0.000 | 0.000 | 0.000 | 350.377 | 164.267 | 54.157 | 3.033 | 0.041 | 0.012 |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 206.152 | 163.236 | 41.102 | 3.971 | 0.027 | 0.013 |
| reasoning_arithmetic | 2048 | 0.000 | 0.000 | 0.000 | 350.220 | 224.258 | 52.110 | 4.304 | 0.025 | 0.013 |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 206.854 | 165.952 | 45.622 | 3.638 | 0.036 | 0.014 |
| retrieval_passkey | 2048 | 0.000 | 0.000 | 0.000 | 349.860 | 228.756 | 59.276 | 3.859 | 0.039 | 0.018 |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE |  |  |
| instruction_constraints | 1024 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE |  |  |
| instruction_constraints | 1024 | systems | 0.000 | STATUS: READY\nCOLOR: BLUE |  |  |
| instruction_constraints | 2048 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE |  |  |
| instruction_constraints | 2048 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE |  |  |
| instruction_constraints | 2048 | systems | 0.000 | STATUS: READY\nCOLOR: BLUE |  |  |
| reasoning_arithmetic | 1024 | exact | 0.000 | 48 |  |  |
| reasoning_arithmetic | 1024 | quality | 0.000 | 48 |  |  |
| reasoning_arithmetic | 1024 | systems | 0.000 | 48 |  |  |
| reasoning_arithmetic | 2048 | exact | 0.000 | 48 |  |  |
| reasoning_arithmetic | 2048 | quality | 0.000 | 48 |  |  |
| reasoning_arithmetic | 2048 | systems | 0.000 | 48 |  |  |
| retrieval_passkey | 1024 | exact | 0.000 | RIVER-58142 |  |  |
| retrieval_passkey | 1024 | quality | 0.000 | RIVER-58142 |  |  |
| retrieval_passkey | 1024 | systems | 0.000 | RIVER-58142 |  |  |
| retrieval_passkey | 2048 | exact | 0.000 | RIVER-58142 |  |  |
| retrieval_passkey | 2048 | quality | 0.000 | RIVER-58142 |  |  |
| retrieval_passkey | 2048 | systems | 0.000 | RIVER-58142 |  |  |
