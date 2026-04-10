| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 | 5632.0 |
| linear_softmax | 160 | 0.900 | 0.944 | 149.2 | 3648.1 | 0.0 |
| linear_softmax_compression_weighted | 160 | 0.894 | 0.938 | 150.2 | 3639.3 | 0.0 |
| linear_softmax_compression_calibrated | 160 | 0.762 | 0.769 | 45.8 | 3005.7 | 0.0 |
| candidate_linear_safe | 160 | 0.894 | 1.000 | 598.4 | 4316.9 | 5632.0 |
