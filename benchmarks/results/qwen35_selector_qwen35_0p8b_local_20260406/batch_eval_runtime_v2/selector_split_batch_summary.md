| split | baseline | test_examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| family_reasoning_holdout | candidate_linear_safe | 160 | 0.900 | 1.000 | 563.2 | 4281.7 |
| family_reasoning_holdout | linear_softmax | 160 | 1.000 | 1.000 | 0.0 | 3718.5 |
| family_reasoning_holdout | linear_softmax_compression_calibrated | 160 | 1.000 | 1.000 | 0.0 | 3718.5 |
| family_reasoning_holdout | linear_softmax_compression_weighted | 160 | 1.000 | 1.000 | 0.0 | 3718.5 |
| family_reasoning_holdout | static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 |
| retrieval_layer23_holdout | candidate_linear_safe | 28 | 0.893 | 1.000 | 603.4 | 3367.0 |
| retrieval_layer23_holdout | linear_softmax | 28 | 1.000 | 1.000 | 0.0 | 2763.6 |
| retrieval_layer23_holdout | linear_softmax_compression_calibrated | 28 | 1.000 | 1.000 | 0.0 | 2763.6 |
| retrieval_layer23_holdout | linear_softmax_compression_weighted | 28 | 1.000 | 1.000 | 0.0 | 2763.6 |
| retrieval_layer23_holdout | static_rule | 28 | 0.750 | 0.750 | 0.0 | 1959.0 |
| variant_logic_holdout | candidate_linear_safe | 88 | 0.909 | 1.000 | 512.0 | 4497.2 |
| variant_logic_holdout | linear_softmax | 88 | 1.000 | 1.000 | 0.0 | 3985.2 |
| variant_logic_holdout | linear_softmax_compression_calibrated | 88 | 1.000 | 1.000 | 0.0 | 3985.2 |
| variant_logic_holdout | linear_softmax_compression_weighted | 88 | 1.000 | 1.000 | 0.0 | 3985.2 |
| variant_logic_holdout | static_rule | 88 | 0.727 | 0.841 | 532.8 | 3825.2 |

## Aggregate
| baseline | folds | mean_target_accuracy | std_target_accuracy | mean_safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate_linear_safe | 3 | 0.901 | 0.007 | 1.000 | 559.5 | 4048.7 |
| linear_softmax | 3 | 1.000 | 0.000 | 1.000 | 0.0 | 3489.1 |
| linear_softmax_compression_calibrated | 3 | 1.000 | 0.000 | 1.000 | 0.0 | 3489.1 |
| linear_softmax_compression_weighted | 3 | 1.000 | 0.000 | 1.000 | 0.0 | 3489.1 |
| static_rule | 3 | 0.740 | 0.010 | 0.818 | 364.6 | 3141.2 |
