# Qwen3.5 9B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 1.000 | 1.000 | 1.000 | 204.946 | 129.965 | 42.836 | 3.034 | 0.043 | 0.012 |
| instruction_constraints | 2048 | 1.000 | 1.000 | 1.000 | 350.439 | 166.003 | 54.497 | 3.046 | 0.041 | 0.012 |
| reasoning_arithmetic | 1024 | 1.000 | 1.000 | 1.000 | 210.022 | 172.191 | 41.983 | 4.101 | 0.072 | 0.051 |
| reasoning_arithmetic | 2048 | 1.000 | 1.000 | 1.000 | 356.973 | 233.687 | 54.198 | 4.312 | 0.077 | 0.046 |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 1.000 | 205.771 | 166.330 | 45.144 | 3.684 | 0.036 | 0.014 |
| retrieval_passkey | 2048 | 1.000 | 1.000 | 1.000 | 349.153 | 230.647 | 59.217 | 3.895 | 0.039 | 0.018 |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | exact | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 1024 | quality | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 1024 | systems | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | exact | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | quality | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| instruction_constraints | 2048 | systems | 1.000 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEuser |
| reasoning_arithmetic | 1024 | exact | 1.000 | 48 | 48 | 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | quality | 1.000 | 48 | 48 | 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48 |
| reasoning_arithmetic | 1024 | systems | 1.000 | 48 | 48 | 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | exact | 1.000 | 48 | 48 | 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | quality | 1.000 | 48 | 48 | 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48 |
| reasoning_arithmetic | 2048 | systems | 1.000 | 48 | 48 | 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48\nFINAL: 48 |
| retrieval_passkey | 1024 | exact | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\n\nBackground memo about |
| retrieval_passkey | 1024 | quality | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\n\nBackground memo about |
| retrieval_passkey | 1024 | systems | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\n\nBackground memo about |
| retrieval_passkey | 2048 | exact | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\nBackground memo about |
| retrieval_passkey | 2048 | quality | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\nBackground memo about |
| retrieval_passkey | 2048 | systems | 1.000 | RIVER-58142 | RIVER-58142 | RIVER-58142\nBackground memo about |
