# Qwen3.5 9B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 219.109 | 147.540 | 45.447 | 3.246 | 0.043 | 0.012 |
| instruction_constraints | 2048 | 0.000 | 0.000 | 0.000 | 360.436 | 185.837 | 57.789 | 3.216 | 0.041 | 0.012 |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 216.322 | 187.638 | 43.219 | 4.342 | 0.027 | 0.013 |
| reasoning_arithmetic | 2048 | 0.000 | 0.000 | 0.000 | 361.989 | 256.837 | 55.628 | 4.617 | 0.025 | 0.013 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 213.907 | 191.098 | 48.423 | 3.946 | 0.036 | 0.014 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 1.000 | 361.022 | 261.909 | 63.116 | 4.150 | 0.039 | 0.018 |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 1024 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 1024 | systems | 0.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | systems | 0.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
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
