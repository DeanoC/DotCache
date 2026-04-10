# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 240.056 | 673.338 | 0.000 | - | 0.989 | - | 0.215 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 306.434 | 854.921 | 0.000 | - | 0.988 | - | 0.223 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 268.004 | 666.122 | 0.000 | - | 0.998 | - | 0.215 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 346.015 | 945.620 | 0.000 | - | 0.996 | - | 0.187 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 288.144 | 663.208 | 0.000 | - | 0.997 | - | 0.216 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 348.955 | 852.743 | 0.000 | - | 0.997 | - | 0.201 | - |

## Sample Outputs

| task | prompt_length | profile | success | cap_hit | generated_tokens | decode_steps | expected | generated_first_line_cleaned | generated_text_cleaned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | exact | 0.000 | yes | 8 | 8 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\nSTATUS: READY\nCOLOR: |
| instruction_constraints | 512 | quality | 0.000 | yes | 8 | 8 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\nSTATUS: READY\nCOLOR: |
| instruction_constraints | 1024 | exact | 0.000 | yes | 8 | 8 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\nSTATUS: READY\nCOLOR: |
| instruction_constraints | 1024 | quality | 0.000 | yes | 8 | 8 | STATUS: READY\nCOLOR: BLUE | </think> | </think>\nSTATUS: READY\nCOLOR: |
| reasoning_arithmetic | 512 | exact | 0.000 | yes | 16 | 16 | 48 | integer> <think> 1. **Analyze the | integer> <think> 1. **Analyze the |
| reasoning_arithmetic | 512 | quality | 0.000 | yes | 16 | 16 | 48 | integer> <think> 1. **Analyze the | integer> <think> 1. **Analyze the |
| reasoning_arithmetic | 1024 | exact | 0.000 | yes | 16 | 16 | 48 | integer> <think> 1. **Analyze the | integer> <think> 1. **Analyze the |
| reasoning_arithmetic | 1024 | quality | 0.000 | yes | 16 | 16 | 48 | integer> <think> 1. **Analyze the | integer> <think> 1. **Analyze the |
| retrieval_passkey | 512 | exact | 0.000 | yes | 16 | 16 | RIVER-58142 | IVER-58142 </think> RIVER-581 | IVER-58142 </think> RIVER-581 |
| retrieval_passkey | 512 | quality | 0.000 | yes | 16 | 16 | RIVER-58142 | IVER-58142 </think> RIVER-581 | IVER-58142 </think> RIVER-581 |
| retrieval_passkey | 1024 | exact | 0.000 | yes | 16 | 16 | RIVER-58142 | IVER-58142 </think> RIVER-581 | IVER-58142 </think> RIVER-581 |
| retrieval_passkey | 1024 | quality | 0.000 | yes | 16 | 16 | RIVER-58142 | IVER-58142 </think> RIVER-581 | IVER-58142 </think> RIVER-581 |
