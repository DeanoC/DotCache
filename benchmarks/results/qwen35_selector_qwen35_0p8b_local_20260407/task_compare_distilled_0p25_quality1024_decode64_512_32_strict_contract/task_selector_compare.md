# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 0.000 | 936.358 | 0.000 | - | 1.026 | - | 0.293 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 0.000 | 1175.774 | 0.000 | - | 1.003 | - | 0.539 | - |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 0.000 | 1171.559 | 0.000 | - | 1.021 | - | 0.246 | - |

## Sample Outputs

| task | prompt_length | profile | success | cap_hit | generated_tokens | decode_steps | expected | generated_first_line_cleaned | generated_text_cleaned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | quality | 0.000 | yes | 32 | 32 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE Answer\nSTATUS: READY\nCOLOR: BLUE\nThe only visible output must be exactly two |
| reasoning_arithmetic | 1024 | quality | 0.000 | yes | 512 | 512 | 48 | 17 + 26 - 9 + 14 = 44 | 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - 9 + 14 = 44\nFINAL: 17 + 26 - |
| retrieval_passkey | 1024 | quality | 0.000 | yes | 64 | 64 | RIVER-58142 | IVER-58142 RIVER-58142 Step-by-Step Solution: 1. **Analyze the Request:** * **Input:** A series of repeated text strings ("Background memo about permit backlogs, bridge closures, zoning appeals, and | IVER-58142 RIVER-58142 Step-by-Step Solution: 1. **Analyze the Request:** * **Input:** A series of repeated text strings ("Background memo about permit backlogs, bridge closures, zoning appeals, and |
