# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 285.540 | 107.970 | 0.000 | - | 0.989 | - | 0.147 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 323.595 | 114.181 | 0.000 | - | 0.983 | - | 0.164 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 286.277 | 135.156 | 0.000 | - | 1.003 | - | 0.197 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 346.623 | 124.910 | 0.000 | - | 1.002 | - | 0.167 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 297.079 | 111.443 | 0.000 | - | 0.997 | - | 0.162 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 327.102 | 125.387 | 0.000 | - | 0.997 | - | 0.152 | - |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\n\nSTATUS: READY\nCOLOR: |
| instruction_constraints | 512 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\n\nSTATUS: READY\nCOLOR: |
| instruction_constraints | 1024 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\n\nSTATUS: READY\nCOLOR: |
| instruction_constraints | 1024 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\n\nSTATUS: READY\nCOLOR: |
| reasoning_arithmetic | 512 | exact | 0.000 | 48 | integer> | integer>\n\n<think>\nThinking Process:\n\n1.  **Analyze the |
| reasoning_arithmetic | 512 | quality | 0.000 | 48 | integer> | integer>\n\n<think>\nThinking Process:\n\n1.  **Analyze the |
| reasoning_arithmetic | 1024 | exact | 0.000 | 48 | integer> | integer>\n\n<think>\nThinking Process:\n\n1.  **Analyze the |
| reasoning_arithmetic | 1024 | quality | 0.000 | 48 | integer> | integer>\n\n<think>\nThinking Process:\n\n1.  **Analyze the |
| retrieval_passkey | 512 | exact | 0.000 | RIVER-58142 | IVER-58142 | IVER-58142\n</think>\n\nRIVER-581 |
| retrieval_passkey | 512 | quality | 0.000 | RIVER-58142 | IVER-58142 | IVER-58142\n</think>\n\nRIVER-581 |
| retrieval_passkey | 1024 | exact | 0.000 | RIVER-58142 | IVER-58142 | IVER-58142\n</think>\n\nRIVER-581 |
| retrieval_passkey | 1024 | quality | 0.000 | RIVER-58142 | IVER-58142 | IVER-58142\n</think>\n\nRIVER-581 |
