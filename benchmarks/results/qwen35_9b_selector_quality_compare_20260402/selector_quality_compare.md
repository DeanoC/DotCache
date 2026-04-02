# Qwen3.5 Selector Quality Compare

## Metrics

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Token agreement | Logit RMSE | Logit max abs | Replay ctx max abs | Replay out max abs | M3 frac | Profile | Logit offset |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1024 | exact | 209.523 | 216.164 | 68.209 | 1.000 | 0.530 | 9.104 | 8.431 | 6.281 | 0.00 | quality | 0.000 |
| 1024 | quality | 72.838 | 91.264 | 67.908 | 1.000 | 0.470 | 7.365 | 7.971 | 11.199 | 95.41 | quality | 0.000 |
| 1024 | systems | 44.032 | 67.723 | 66.743 | 1.000 | 0.467 | 7.244 | 7.961 | 11.336 | 98.83 | systems | 2.000 |
| 2048 | exact | 354.765 | 361.452 | 120.620 | 1.000 | 0.158 | 1.258 | 2.270 | 3.086 | 0.00 | quality | 0.000 |
| 2048 | quality | 87.594 | 88.864 | 119.781 | 1.000 | 0.278 | 3.734 | 5.212 | 11.246 | 96.46 | quality | 0.000 |
| 2048 | systems | 48.313 | 48.653 | 118.703 | 1.000 | 0.272 | 3.727 | 5.205 | 11.223 | 99.85 | systems | 2.000 |

## Tradeoff

| Context | Quality vs Exact speedup | Systems vs Exact speedup | Systems vs Quality speedup | Systems - Quality token agreement | Systems - Quality logit RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2.877 | 4.758 | 1.654 | 0.000 | -0.003 |
| 2048 | 4.050 | 7.343 | 1.813 | 0.000 | -0.005 |
