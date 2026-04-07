| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 13 | 0.615 | 0.769 | 704.0 | 5350.2 | 3731.2 |
| linear_softmax | 13 | 1.000 | 1.000 | 0.0 | 5133.6 | 0.0 |
| linear_softmax_compression_weighted | 13 | 1.000 | 1.000 | 0.0 | 5133.6 | 0.0 |
| linear_softmax_compression_calibrated | 13 | 1.000 | 1.000 | 0.0 | 5133.6 | 0.0 |
| candidate_linear_safe | 13 | 0.923 | 1.000 | 433.2 | 5566.8 | 2252.8 |
