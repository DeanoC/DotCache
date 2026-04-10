# Qwen Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_ppl_ratio | systems_ppl_ratio | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | 0.000 | 0.000 | 0.000 | 297.656 | 184.820 | 0.000 | - | 1.056 | - | 0.602 | - |
| instruction_constraints | 1024 | 1.000 | 1.000 | 0.000 | 357.410 | 240.669 | 0.000 | - | 1.024 | - | 0.400 | - |
| reasoning_arithmetic | 512 | 0.000 | 0.000 | 0.000 | 301.719 | 220.840 | 0.000 | - | 1.030 | - | 0.464 | - |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 341.198 | 239.490 | 0.000 | - | 1.038 | - | 0.458 | - |
| retrieval_passkey | 512 | 1.000 | 1.000 | 0.000 | 294.461 | 351.483 | 0.000 | - | 1.014 | - | 0.334 | - |
| retrieval_passkey | 1024 | 1.000 | 1.000 | 0.000 | 331.995 | 236.797 | 0.000 | - | 1.043 | - | 0.317 | - |

## Sample Outputs

| task | prompt_length | profile | success | cap_hit | generated_tokens | decode_steps | expected | generated_first_line_cleaned | generated_text_cleaned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 512 | exact | 0.000 | yes | 16 | 16 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEFollow these instructions exactly. |
| instruction_constraints | 512 | quality | 0.000 | yes | 16 | 16 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUEFollow these instructions exactly. |
| instruction_constraints | 1024 | exact | 1.000 | yes | 16 | 16 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE\nFollow these instructions exactly. |
| instruction_constraints | 1024 | quality | 1.000 | yes | 16 | 16 | STATUS: READY\nCOLOR: BLUE | STATUS: READY | STATUS: READY\nCOLOR: BLUE\nFollow these instructions exactly. |
| reasoning_arithmetic | 512 | exact | 0.000 | yes | 64 | 64 | 48 | integer> 1. **Analyze the Request:** * Input text: A series of repetitions of aArchived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "A clerk solves a budget worksheet." and a | integer> 1. **Analyze the Request:** * Input text: A series of repetitions of aArchived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "A clerk solves a budget worksheet." and a |
| reasoning_arithmetic | 512 | quality | 0.000 | yes | 64 | 64 | 48 | integer> 1. **Analyze the Request:** * Input text: A series of repetitions of aArchived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "A clerk solves a budget worksheet." and a | integer> 1. **Analyze the Request:** * Input text: A series of repetitions of aArchived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "A clerk solves a budget worksheet." and a |
| reasoning_arithmetic | 1024 | exact | 0.000 | yes | 64 | 64 | 48 | integer> 1. **Analyze the Request:** * Input text: A series of repetitions phrases ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "Aived clerk solves a budget worksheet." and | integer> 1. **Analyze the Request:** * Input text: A series of repetitions phrases ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "Aived clerk solves a budget worksheet." and |
| reasoning_arithmetic | 1024 | quality | 0.000 | yes | 64 | 64 | 48 | integer> 1. **Analyze the Request:** * Input text: A series of repeated phrases ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "Aived clerk solves a budget worksheet." and | integer> 1. **Analyze the Request:** * Input text: A series of repeated phrases ("Archived finance notes mention approvals, invoices, compliance dates, and transport budgets across several quarters." followed by a specific "Aived clerk solves a budget worksheet." and |
| retrieval_passkey | 512 | exact | 1.000 | yes | 32 | 32 | RIVER-58142 | IVER-58142 RIVER-58142 Step-by-stepStep Solution 1. **An | IVER-58142 RIVER-58142 Step-by-stepStep Solution 1. **An |
| retrieval_passkey | 512 | quality | 1.000 | yes | 32 | 32 | RIVER-58142 | IVER-58142 RIVER-58142 Step-by-stepStep Solution 1. **An | IVER-58142 RIVER-58142 Step-by-stepStep Solution 1. **An |
| retrieval_passkey | 1024 | exact | 1.000 | yes | 32 | 32 | RIVER-58142 | IVER-58142 RIVER-58142 Response-by-stepStep Solution: 1. **An | IVER-58142 RIVER-58142 Response-by-stepStep Solution: 1. **An |
| retrieval_passkey | 1024 | quality | 1.000 | yes | 32 | 32 | RIVER-58142 | IVER-58142 RIVER-58142 Response-by-stepStep Solution: 1. **An | IVER-58142 RIVER-58142 Response-by-stepStep Solution: 1. **An |
