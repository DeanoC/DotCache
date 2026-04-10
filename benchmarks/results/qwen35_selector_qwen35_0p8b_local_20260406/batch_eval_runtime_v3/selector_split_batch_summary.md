| split | baseline | test_examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| family_reasoning_holdout | candidate_linear_safe | 160 | 0.900 | 1.000 | 563.2 | 4281.7 |
| family_reasoning_holdout | linear_softmax | 160 | 0.906 | 0.944 | 111.9 | 3612.9 |
| family_reasoning_holdout | linear_softmax_compression_calibrated | 160 | 0.719 | 0.719 | 0.0 | 2873.7 |
| family_reasoning_holdout | linear_softmax_compression_weighted | 160 | 0.881 | 0.912 | 106.1 | 3507.3 |
| family_reasoning_holdout | static_rule | 160 | 0.744 | 0.863 | 561.2 | 3639.3 |
| retrieval_layer23_holdout | candidate_linear_safe | 28 | 0.893 | 1.000 | 603.4 | 3367.0 |
| retrieval_layer23_holdout | linear_softmax | 28 | 0.964 | 1.000 | 201.1 | 2964.7 |
| retrieval_layer23_holdout | linear_softmax_compression_calibrated | 28 | 0.786 | 0.786 | 0.0 | 2009.3 |
| retrieval_layer23_holdout | linear_softmax_compression_weighted | 28 | 1.000 | 1.000 | 0.0 | 2763.6 |
| retrieval_layer23_holdout | static_rule | 28 | 0.750 | 0.750 | 0.0 | 1959.0 |
| variant_logic_holdout | candidate_linear_safe | 88 | 0.909 | 1.000 | 512.0 | 4497.2 |
| variant_logic_holdout | linear_softmax | 88 | 0.898 | 0.943 | 118.7 | 3921.2 |
| variant_logic_holdout | linear_softmax_compression_calibrated | 88 | 0.750 | 0.750 | 0.0 | 3249.2 |
| variant_logic_holdout | linear_softmax_compression_weighted | 88 | 0.886 | 0.920 | 104.3 | 3873.2 |
| variant_logic_holdout | static_rule | 88 | 0.727 | 0.841 | 532.8 | 3825.2 |

## Aggregate
| baseline | folds | mean_target_accuracy | std_target_accuracy | mean_safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate_linear_safe | 3 | 0.901 | 0.007 | 1.000 | 559.5 | 4048.7 |
| linear_softmax | 3 | 0.923 | 0.030 | 0.962 | 143.9 | 3499.6 |
| linear_softmax_compression_calibrated | 3 | 0.751 | 0.027 | 0.751 | 0.0 | 2710.7 |
| linear_softmax_compression_weighted | 3 | 0.923 | 0.055 | 0.944 | 70.1 | 3381.4 |
| static_rule | 3 | 0.740 | 0.010 | 0.818 | 364.6 | 3141.2 |
