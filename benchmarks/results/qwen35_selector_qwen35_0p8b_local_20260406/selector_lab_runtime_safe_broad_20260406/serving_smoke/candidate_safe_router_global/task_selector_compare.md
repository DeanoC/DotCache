# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 334.413 | 841.171 | 0.000 | - | 1.000 | - | 0.242 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 369.306 | 1251.777 | 0.000 | - | 1.001 | - | 0.249 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 269.471 | 827.760 | 0.000 | - | 0.998 | - | 0.229 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 360.997 | 1213.047 | 0.000 | - | 0.997 | - | 0.202 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 286.699 | 949.297 | 0.000 | - | 0.998 | - | 0.229 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 363.080 | 1077.395 | 0.000 | - | 0.997 | - | 0.215 | - |

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
