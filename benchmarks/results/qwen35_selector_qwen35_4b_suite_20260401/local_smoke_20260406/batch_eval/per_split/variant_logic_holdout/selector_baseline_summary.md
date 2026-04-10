| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 63 | 0.683 | 0.810 | 386.5 | 3716.4 | 1408.0 |
| linear_softmax | 63 | 1.000 | 1.000 | 0.0 | 4074.0 | 0.0 |
| linear_softmax_compression_weighted | 63 | 1.000 | 1.000 | 0.0 | 4074.0 | 0.0 |
| linear_softmax_compression_calibrated | 63 | 0.984 | 0.984 | 0.0 | 4051.6 | 0.0 |
| candidate_linear_safe | 63 | 0.889 | 0.968 | 461.6 | 4651.0 | 5632.0 |
