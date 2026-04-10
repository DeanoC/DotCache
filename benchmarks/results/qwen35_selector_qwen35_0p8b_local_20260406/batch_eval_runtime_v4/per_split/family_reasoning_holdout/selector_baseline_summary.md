| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 | 5632.0 |
| linear_softmax | 160 | 0.900 | 0.944 | 149.2 | 3648.1 | 0.0 |
| linear_softmax_compression_weighted | 160 | 0.900 | 0.956 | 220.9 | 3736.1 | 1408.0 |
| linear_softmax_compression_calibrated | 160 | 0.838 | 0.844 | 41.7 | 3216.9 | 0.0 |
| candidate_linear_safe | 160 | 0.894 | 1.000 | 598.4 | 4316.9 | 5632.0 |
