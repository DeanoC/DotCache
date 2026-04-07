# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 319.576 | 973.857 | 0.000 | - | 0.993 | - | 0.221 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 335.436 | 1484.513 | 0.000 | - | 0.996 | - | 0.233 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 293.890 | 924.979 | 0.000 | - | 0.997 | - | 0.216 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 347.097 | 1380.830 | 0.000 | - | 0.998 | - | 0.190 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 306.103 | 942.294 | 0.000 | - | 0.997 | - | 0.221 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 367.828 | 1105.213 | 0.000 | - | 0.997 | - | 0.204 | - |

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
