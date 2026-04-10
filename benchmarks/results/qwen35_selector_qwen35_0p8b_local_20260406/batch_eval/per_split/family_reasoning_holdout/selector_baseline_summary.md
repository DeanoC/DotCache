| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 | 5632.0 |
| linear_softmax | 160 | 1.000 | 1.000 | 0.0 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | 160 | 1.000 | 1.000 | 0.0 | 3718.5 | 0.0 |
| linear_softmax_compression_calibrated | 160 | 0.994 | 0.994 | 0.0 | 3709.7 | 0.0 |
| candidate_linear_safe | 160 | 0.906 | 1.000 | 528.0 | 4246.5 | 5632.0 |
