# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher | ok | row_multiclass | runtime_safe | global | yes | yes | 0.928 | 0.889 | 3424.8 | 85.1 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher | family_reasoning_holdout | 0.881 | 0.925 | 3621.7 | 152.2 |
| linear_softmax_distilled_mlp_teacher | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_distilled_mlp_teacher | variant_logic_holdout | 0.898 | 0.932 | 3889.2 | 103.0 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher | reasoning | 0.889 | 0.928 | 3755.5 |
| linear_softmax_distilled_mlp_teacher | retrieval | 1.000 | 1.000 | 2763.6 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher | arithmetic | 0.875 | 0.931 | 3314.4 |
| linear_softmax_distilled_mlp_teacher | logic | 0.892 | 0.926 | 3881.2 |
| linear_softmax_distilled_mlp_teacher | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_distilled_mlp_teacher | transcript | 1.000 | 1.000 | 3027.6 |
