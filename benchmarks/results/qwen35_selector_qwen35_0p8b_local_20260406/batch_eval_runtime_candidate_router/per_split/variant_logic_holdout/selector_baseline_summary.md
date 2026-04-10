| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 88 | 0.727 | 0.841 | 532.8 | 3825.2 | 5632.0 |
| linear_softmax | 88 | 0.909 | 0.943 | 101.8 | 3905.2 | 0.0 |
| linear_softmax_compression_weighted | 88 | 0.909 | 0.943 | 101.8 | 3905.2 | 0.0 |
| linear_softmax_compression_calibrated | 88 | 0.761 | 0.773 | 82.8 | 3345.2 | 0.0 |
| candidate_linear_safe | 88 | 0.909 | 1.000 | 512.0 | 4497.2 | 5632.0 |
| candidate_safe_router | 88 | 0.909 | 1.000 | 512.0 | 4497.2 | 5632.0 |
