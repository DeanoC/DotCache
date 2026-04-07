| baseline | examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes | p95_safe_bytes_regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static_rule | 28 | 0.750 | 0.750 | 0.0 | 1959.0 | 0.0 |
| linear_softmax | 28 | 0.964 | 1.000 | 201.1 | 2964.7 | 0.0 |
| linear_softmax_compression_weighted | 28 | 0.964 | 1.000 | 201.1 | 2964.7 | 0.0 |
| linear_softmax_compression_calibrated | 28 | 0.893 | 0.893 | 0.0 | 2311.0 | 0.0 |
| candidate_linear_safe | 28 | 0.893 | 1.000 | 603.4 | 3367.0 | 5632.0 |
