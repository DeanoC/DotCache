# Qwen3.5 9B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 1.000 | 211.674 | 137.894 | 44.195 | 3.120 | 0.043 | 0.012 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 1.000 | 356.072 | 173.162 | 55.273 | 3.133 | 0.041 | 0.012 |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 211.229 | 174.250 | 41.644 | 4.184 | 0.027 | 0.013 |
| reasoning_arithmetic | 2048 | 0.000 | 0.000 | 0.000 | 356.054 | 238.482 | 53.339 | 4.471 | 0.025 | 0.013 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 211.135 | 176.339 | 47.080 | 3.746 | 0.036 | 0.014 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 1.000 | 356.135 | 245.086 | 60.399 | 4.058 | 0.039 | 0.018 |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | exact | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 1024 | quality | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 1024 | systems | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | exact | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | quality | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | systems | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| reasoning_arithmetic | 1024 | exact | 0.000 | 48 | <think> | <think>\nThinking Process:\n\n1.  **An |
| reasoning_arithmetic | 1024 | quality | 0.000 | 48 | <think> | <think>\nThinking Process:\n\n1.  **An |
| reasoning_arithmetic | 1024 | systems | 0.000 | 48 | <think> | <think>\nThinking Process:\n\n1.  **An |
| reasoning_arithmetic | 2048 | exact | 0.000 | 48 | <think> | <think>\nThinking Process:\n\n1.  **An |
| reasoning_arithmetic | 2048 | quality | 0.000 | 48 | <think> | <think>\nThinking Process:\n\n1.  **An |
| reasoning_arithmetic | 2048 | systems | 0.000 | 48 | <think> | <think>\nThinking Process:\n\n1.  **An |
| retrieval_passkey | 1024 | exact | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\n\nBackground memo about |
| retrieval_passkey | 1024 | quality | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\n\nBackground memo about |
| retrieval_passkey | 1024 | systems | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\n\nBackground memo about |
| retrieval_passkey | 2048 | exact | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\nBackground memo about |
| retrieval_passkey | 2048 | quality | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\nBackground memo about |
| retrieval_passkey | 2048 | systems | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\nBackground memo about |
