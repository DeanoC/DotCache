| split | baseline | test_examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| family_reasoning_holdout | candidate_linear_safe | 160 | 0.894 | 1.000 | 598.4 | 4316.9 |
| family_reasoning_holdout | linear_softmax | 160 | 0.900 | 0.944 | 149.2 | 3648.1 |
| family_reasoning_holdout | linear_softmax_compression_calibrated | 160 | 0.838 | 0.844 | 41.7 | 3216.9 |
| family_reasoning_holdout | linear_softmax_compression_weighted | 160 | 0.900 | 0.956 | 220.9 | 3736.1 |
| family_reasoning_holdout | static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 |
| retrieval_layer23_holdout | candidate_linear_safe | 28 | 0.893 | 1.000 | 603.4 | 3367.0 |
| retrieval_layer23_holdout | linear_softmax | 28 | 0.964 | 1.000 | 201.1 | 2964.7 |
| retrieval_layer23_holdout | linear_softmax_compression_calibrated | 28 | 0.893 | 0.893 | 0.0 | 2311.0 |
| retrieval_layer23_holdout | linear_softmax_compression_weighted | 28 | 0.964 | 1.000 | 201.1 | 2964.7 |
| retrieval_layer23_holdout | static_rule | 28 | 0.750 | 0.750 | 0.0 | 1959.0 |
| variant_logic_holdout | candidate_linear_safe | 88 | 0.909 | 1.000 | 512.0 | 4497.2 |
| variant_logic_holdout | linear_softmax | 88 | 0.909 | 0.943 | 101.8 | 3905.2 |
| variant_logic_holdout | linear_softmax_compression_calibrated | 88 | 0.830 | 0.841 | 76.1 | 3489.2 |
| variant_logic_holdout | linear_softmax_compression_weighted | 88 | 0.898 | 0.955 | 184.4 | 4001.2 |
| variant_logic_holdout | static_rule | 88 | 0.727 | 0.841 | 532.8 | 3825.2 |

## Aggregate
| baseline | folds | mean_target_accuracy | std_target_accuracy | mean_safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate_linear_safe | 3 | 0.899 | 0.007 | 1.000 | 571.3 | 4060.4 |
| linear_softmax | 3 | 0.924 | 0.028 | 0.962 | 150.7 | 3506.0 |
| linear_softmax_compression_calibrated | 3 | 0.853 | 0.028 | 0.859 | 39.3 | 3005.7 |
| linear_softmax_compression_weighted | 3 | 0.921 | 0.031 | 0.970 | 202.1 | 3567.4 |
| static_rule | 3 | 0.740 | 0.010 | 0.818 | 364.6 | 3141.2 |
