| split | baseline | test_examples | target_accuracy | safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| retrieval_layer23_holdout | candidate_linear_safe | 13 | 0.923 | 1.000 | 433.2 | 5566.8 |
| retrieval_layer23_holdout | linear_softmax | 13 | 1.000 | 1.000 | 0.0 | 5133.6 |
| retrieval_layer23_holdout | linear_softmax_compression_calibrated | 13 | 1.000 | 1.000 | 0.0 | 5133.6 |
| retrieval_layer23_holdout | linear_softmax_compression_weighted | 13 | 1.000 | 1.000 | 0.0 | 5133.6 |
| retrieval_layer23_holdout | static_rule | 13 | 0.615 | 0.769 | 704.0 | 5350.2 |
| variant_logic_holdout | candidate_linear_safe | 63 | 0.889 | 0.968 | 461.6 | 4651.0 |
| variant_logic_holdout | linear_softmax | 63 | 1.000 | 1.000 | 0.0 | 4074.0 |
| variant_logic_holdout | linear_softmax_compression_calibrated | 63 | 0.984 | 0.984 | 0.0 | 4051.6 |
| variant_logic_holdout | linear_softmax_compression_weighted | 63 | 1.000 | 1.000 | 0.0 | 4074.0 |
| variant_logic_holdout | static_rule | 63 | 0.683 | 0.810 | 386.5 | 3716.4 |
| family_reasoning_holdout | candidate_linear_safe | 122 | 0.893 | 0.984 | 516.3 | 4786.2 |
| family_reasoning_holdout | linear_softmax | 122 | 1.000 | 1.000 | 0.0 | 4211.3 |
| family_reasoning_holdout | linear_softmax_compression_calibrated | 122 | 0.992 | 0.992 | 0.0 | 4199.7 |
| family_reasoning_holdout | linear_softmax_compression_weighted | 122 | 1.000 | 1.000 | 0.0 | 4211.3 |
| family_reasoning_holdout | static_rule | 122 | 0.721 | 0.836 | 400.3 | 4038.2 |

## Aggregate
| baseline | folds | mean_target_accuracy | std_target_accuracy | mean_safe_prediction_rate | mean_safe_bytes_regret | mean_predicted_total_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate_linear_safe | 3 | 0.902 | 0.015 | 0.984 | 470.4 | 5001.4 |
| linear_softmax | 3 | 1.000 | 0.000 | 1.000 | 0.0 | 4473.0 |
| linear_softmax_compression_calibrated | 3 | 0.992 | 0.006 | 0.992 | 0.0 | 4461.7 |
| linear_softmax_compression_weighted | 3 | 1.000 | 0.000 | 1.000 | 0.0 | 4473.0 |
| static_rule | 3 | 0.673 | 0.044 | 0.805 | 496.9 | 4368.3 |
