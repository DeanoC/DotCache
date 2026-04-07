# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 304.362 | 921.233 | 0.000 | - | 1.000 | - | 0.242 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 348.873 | 1302.699 | 0.000 | - | 1.001 | - | 0.249 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 287.550 | 913.029 | 0.000 | - | 0.998 | - | 0.229 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 359.083 | 1185.838 | 0.000 | - | 0.997 | - | 0.202 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 301.350 | 981.524 | 0.000 | - | 0.998 | - | 0.229 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 363.635 | 1045.330 | 0.000 | - | 0.997 | - | 0.215 | - |

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
