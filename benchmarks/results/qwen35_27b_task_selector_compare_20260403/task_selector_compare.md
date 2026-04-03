# Qwen3.5 27B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 431.617 | 360.908 | 95.196 | 3.791 | 0.021 | 0.014 |
| instruction_constraints | 2048 | 0.000 | 0.000 | 0.000 | 717.085 | 492.480 | 123.007 | 4.004 | 0.019 | 0.013 |
| reasoning_arithmetic | 1024 | 1.000 | 1.000 | 1.000 | 444.524 | 369.360 | 106.314 | 3.474 | 0.043 | 0.030 |
| reasoning_arithmetic | 2048 | 1.000 | 1.000 | 1.000 | 737.137 | 526.976 | 130.868 | 4.027 | 0.037 | 0.029 |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 429.187 | 350.884 | 99.116 | 3.540 | 0.018 | 0.009 |
| retrieval_passkey | 2048 | 0.000 | 0.000 | 0.000 | 714.466 | 526.836 | 123.895 | 4.252 | 0.019 | 0.010 |

## Sample Outputs

| task | prompt_length | profile | success | expected | generated_first_line | generated_text |
| --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE | <think> | <think>\nThinking Process:\n\n1.  **An |
| instruction_constraints | 1024 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE | <think> | <think>\nThinking Process:\n\n1.  **An |
| instruction_constraints | 1024 | systems | 0.000 | STATUS: READY\nCOLOR: BLUE | <think> | <think>\nThinking Process:\n\n1.  **An |
| instruction_constraints | 2048 | exact | 0.000 | STATUS: READY\nCOLOR: BLUE | <think> | <think>\nThinking Process:\n\n1.  **An |
| instruction_constraints | 2048 | quality | 0.000 | STATUS: READY\nCOLOR: BLUE | <think> | <think>\nThinking Process:\n\n1.  **An |
| instruction_constraints | 2048 | systems | 0.000 | STATUS: READY\nCOLOR: BLUE | <think> | <think>\nThinking Process:\n\n1.  **An |
| reasoning_arithmetic | 1024 | exact | 1.000 | 48 | 48 | 48\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A large block of repeated text ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters.") followed by a specific instruction ("A clerk solves a budget worksheet. |
| reasoning_arithmetic | 1024 | quality | 1.000 | 48 | 48 | 48\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A large block of repeated text ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters.") followed by a specific instruction ("A clerk solves a budget worksheet. |
| reasoning_arithmetic | 1024 | systems | 1.000 | 48 | 48 | 48\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A large block of repeated text ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters.") followed by a specific instruction ("A clerk solves a budget worksheet. |
| reasoning_arithmetic | 2048 | exact | 1.000 | 48 | 48 | 48\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A large block of repeated text ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters.") followed by a specific instruction ("A clerk solves a budget worksheet. |
| reasoning_arithmetic | 2048 | quality | 1.000 | 48 | 48 | 48\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A large block of repeated text ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters.") followed by a specific instruction ("A clerk solves a budget worksheet. |
| reasoning_arithmetic | 2048 | systems | 1.000 | 48 | 48 | 48\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A large block of repeated text ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters.") followed by a specific instruction ("A clerk solves a budget worksheet. |
| retrieval_passkey | 1024 | exact | 0.000 | RIVER-58142 | <think> | <think>\nThinking Process:\n\n1.  **An |
| retrieval_passkey | 1024 | quality | 0.000 | RIVER-58142 | <think> | <think>\nThinking Process:\n\n1.  **An |
| retrieval_passkey | 1024 | systems | 0.000 | RIVER-58142 | <think> | <think>\nThinking Process:\n\n1.  **An |
| retrieval_passkey | 2048 | exact | 0.000 | RIVER-58142 | <think> | <think>\nThinking Process:\n\n1.  **An |
| retrieval_passkey | 2048 | quality | 0.000 | RIVER-58142 | <think> | <think>\nThinking Process:\n\n1.  **An |
| retrieval_passkey | 2048 | systems | 0.000 | RIVER-58142 | <think> | <think>\nThinking Process:\n\n1.  **An |
