| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 88 | 0.727 | 0.841 | 532.8 | 3825.2 | 5632.0 |
| linear_softmax | 88 | 1.000 | 1.000 | 0.0 | 3985.2 | 0.0 |
| linear_softmax_compression_weighted | 88 | 1.000 | 1.000 | 0.0 | 3985.2 | 0.0 |
| linear_softmax_compression_calibrated | 88 | 1.000 | 1.000 | 0.0 | 3985.2 | 0.0 |
| candidate_linear_safe | 88 | 0.909 | 1.000 | 512.0 | 4497.2 | 5632.0 |
