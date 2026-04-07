# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_weighted_dense_control_balanced | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_weighted_dense_control_conservative | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_weighted_dense_control_strong | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| linear_softmax_compression_weighted_dense_control_balanced | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted_dense_control_balanced | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted_dense_control_balanced | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| linear_softmax_compression_weighted_dense_control_conservative | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted_dense_control_conservative | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted_dense_control_conservative | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| linear_softmax_compression_weighted_dense_control_strong | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted_dense_control_strong | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted_dense_control_strong | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_weighted_dense_control_balanced | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted_dense_control_balanced | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_weighted_dense_control_conservative | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted_dense_control_conservative | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_weighted_dense_control_strong | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted_dense_control_strong | retrieval | 1.000 | 1.000 | 2763.6 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_weighted_dense_control_balanced | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted_dense_control_balanced | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted_dense_control_balanced | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted_dense_control_balanced | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_weighted_dense_control_conservative | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted_dense_control_conservative | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted_dense_control_conservative | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted_dense_control_conservative | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_weighted_dense_control_strong | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted_dense_control_strong | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted_dense_control_strong | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted_dense_control_strong | transcript | 1.000 | 1.000 | 3027.6 |
