| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 122 | 0.721 | 0.836 | 400.3 | 4038.2 | 1408.0 |
| linear_softmax | 122 | 1.000 | 1.000 | 0.0 | 4211.3 | 0.0 |
| linear_softmax_compression_weighted | 122 | 1.000 | 1.000 | 0.0 | 4211.3 | 0.0 |
| linear_softmax_compression_calibrated | 122 | 0.992 | 0.992 | 0.0 | 4199.7 | 0.0 |
| candidate_linear_safe | 122 | 0.893 | 0.984 | 516.3 | 4786.2 | 5632.0 |
