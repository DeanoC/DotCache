# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 235.889 | 336.464 | 0.000 | - | 0.990 | - | 0.320 | - |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 356.967 | 358.390 | 0.000 | - | 0.991 | - | 0.292 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 271.193 | 325.858 | 0.000 | - | 0.995 | - | 0.314 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 367.133 | 385.118 | 0.000 | - | 0.996 | - | 0.288 | - |
| retrieval_passkey | 512 | 0.000 | 0.000 | 0.000 | 260.961 | 299.541 | 0.000 | - | 1.000 | - | 0.280 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 377.001 | 380.853 | 0.000 | - | 0.998 | - | 0.257 | - |

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
