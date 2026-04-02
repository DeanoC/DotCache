# Qwen3.5 9B Task Selector Compare

| task | prompt_length | exact_success | quality_success | systems_success | exact_decode_ms | quality_decode_ms | systems_decode_ms | systems_vs_quality_speedup | quality_rmse | systems_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instruction_constraints | 1024 | 0.000 | 0.000 | 0.000 | 209.486 | 134.626 | 43.960 | 3.062 | 0.043 | 0.012 |
| instruction_constraints | 2048 | 0.000 | 0.000 | 0.000 | 353.593 | 170.586 | 55.815 | 3.056 | 0.041 | 0.012 |
| reasoning_arithmetic | 1024 | 0.000 | 0.000 | 0.000 | 208.809 | 170.771 | 41.618 | 4.103 | 0.027 | 0.013 |
| reasoning_arithmetic | 2048 | 0.000 | 0.000 | 0.000 | 355.942 | 234.654 | 53.595 | 4.378 | 0.025 | 0.013 |
| retrieval_passkey | 1024 | 0.000 | 0.000 | 0.000 | 209.469 | 171.789 | 46.772 | 3.673 | 0.036 | 0.014 |
| retrieval_passkey | 2048 | 0.000 | 0.000 | 0.000 | 356.441 | 237.930 | 60.939 | 3.904 | 0.039 | 0.018 |
