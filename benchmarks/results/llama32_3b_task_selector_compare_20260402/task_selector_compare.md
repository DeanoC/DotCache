# Llama 3.2 3B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_logit_max_abs | systems_logit_max_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 1.000 | 120.250 | 59.064 | 59.036 | 1.000 | 0.047 | 0.047 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 1.000 | 162.594 | 72.796 | 76.909 | 0.947 | 0.072 | 0.072 |
| reasoning_arithmetic | 1024 | 1.000 | 1.000 | 1.000 | 121.236 | 58.832 | 59.290 | 0.992 | 0.098 | 0.098 |
| reasoning_arithmetic | 2048 | 1.000 | 1.000 | 1.000 | 162.406 | 72.677 | 72.207 | 1.007 | 0.048 | 0.048 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 120.101 | 58.867 | 60.173 | 0.978 | 0.051 | 0.051 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 1.000 | 164.117 | 72.817 | 75.772 | 0.961 | 0.129 | 0.129 |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | exact | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | quality | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 1024 | systems | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | exact | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | quality | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| instruction_constraints | 2048 | systems | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE |
| reasoning_arithmetic | 1024 | exact | 1.000 | 48 | 48 | 48\nThe final answer is 48. |
| reasoning_arithmetic | 1024 | quality | 1.000 | 48 | 48 | 48\nThe final answer is 48. |
| reasoning_arithmetic | 1024 | systems | 1.000 | 48 | 48 | 48\nThe final answer is 48. |
| reasoning_arithmetic | 2048 | exact | 1.000 | 48 | 48 | 48\nThe final answer is 48. |
| reasoning_arithmetic | 2048 | quality | 1.000 | 48 | 48 | 48\nThe final answer is 48. I'll wait for your confirmation before proceeding.'think silently, but the visible response must end with a final line exactly in the form FINAL: <integer>.\nDo not include any words after the final integer.\nFINAL: 48\nThe final answer is |
| reasoning_arithmetic | 2048 | systems | 1.000 | 48 | 48 | 48\nThe final answer is 48. I'll wait for your confirmation before proceeding.'think silently, but the visible response must end with a final line exactly in the form FINAL: <integer>.\nDo not include any words after the final integer.\nFINAL: 48\nThe final answer is |
| retrieval_passkey | 1024 | exact | 1.000 | RIVER-58142 | RIVER-58142. | RIVER-58142. |
| retrieval_passkey | 1024 | quality | 1.000 | RIVER-58142 | RIVER-58142. | RIVER-58142. |
| retrieval_passkey | 1024 | systems | 1.000 | RIVER-58142 | RIVER-58142. | RIVER-58142. |
| retrieval_passkey | 2048 | exact | 1.000 | RIVER-58142 | RIVER-58142. | RIVER-58142. |
| retrieval_passkey | 2048 | quality | 1.000 | RIVER-58142 | RIVER-58142. | RIVER-58142. |
| retrieval_passkey | 2048 | systems | 1.000 | RIVER-58142 | RIVER-58142. | RIVER-58142. |
