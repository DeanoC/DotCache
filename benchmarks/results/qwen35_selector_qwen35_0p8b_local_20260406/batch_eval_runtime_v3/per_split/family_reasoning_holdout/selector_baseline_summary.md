| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 | 5632.0 |
| linear_softmax | 160 | 0.906 | 0.944 | 111.9 | 3612.9 | 0.0 |
| linear_softmax_compression_weighted | 160 | 0.881 | 0.912 | 106.1 | 3507.3 | 0.0 |
| linear_softmax_compression_calibrated | 160 | 0.719 | 0.719 | 0.0 | 2873.7 | 0.0 |
| candidate_linear_safe | 160 | 0.900 | 1.000 | 563.2 | 4281.7 | 5632.0 |
