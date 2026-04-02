# Qwen3.5 Selector Quality Compare

## Metrics

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Token agreement | Logit RMSE | Logit max abs | Replay ctx max abs | Replay out max abs | M3 frac | Profile | Logit offset |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1024 | exact | 212.435 | 212.975 | 68.154 | 1.000 | 0.530 | 9.104 | 8.431 | 6.281 | 0.00 | systems | 2.000 |
| 1024 | quality | 67.656 | 68.346 | 67.429 | 1.000 | 0.470 | 7.365 | 7.971 | 11.199 | 95.41 | quality | 0.000 |
| 1024 | systems | 45.229 | 63.828 | 67.364 | 1.000 | 0.467 | 7.244 | 7.961 | 11.336 | 98.83 | systems | 2.000 |
| 2048 | exact | 358.189 | 358.766 | 119.699 | 1.000 | 0.158 | 1.258 | 2.270 | 3.086 | 0.00 | systems | 2.000 |
| 2048 | quality | 82.911 | 83.707 | 118.498 | 1.000 | 0.278 | 3.734 | 5.212 | 11.246 | 96.46 | quality | 0.000 |
| 2048 | systems | 49.993 | 50.990 | 118.227 | 1.000 | 0.272 | 3.727 | 5.205 | 11.223 | 99.85 | systems | 2.000 |

## Tradeoff

| Context | Quality vs Exact speedup | Systems vs Exact speedup | Systems vs Quality speedup | Systems - Quality token agreement | Systems - Quality logit RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 3.140 | 4.697 | 1.496 | 0.000 | -0.003 |
| 2048 | 4.320 | 7.165 | 1.658 | 0.000 | -0.005 |
