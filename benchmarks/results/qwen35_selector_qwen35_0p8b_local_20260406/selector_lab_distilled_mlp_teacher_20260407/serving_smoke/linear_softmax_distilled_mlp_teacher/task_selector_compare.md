# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 298.919 | 677.655 | 0.000 | - | 0.990 | - | 0.216 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 292.465 | 888.635 | 0.000 | - | 0.987 | - | 0.223 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 278.152 | 654.732 | 0.000 | - | 0.999 | - | 0.213 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 320.285 | 920.681 | 0.000 | - | 0.996 | - | 0.187 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 277.613 | 602.056 | 0.000 | - | 0.997 | - | 0.217 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 357.030 | 862.488 | 0.000 | - | 0.997 | - | 0.201 | - |

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
