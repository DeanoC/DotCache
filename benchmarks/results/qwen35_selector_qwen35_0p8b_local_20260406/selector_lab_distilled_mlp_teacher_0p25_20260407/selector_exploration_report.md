# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher_0p25 | ok | row_multiclass | runtime_safe | global | yes | yes | 0.932 | 0.893 | 3427.8 | 84.7 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher_0p25 | family_reasoning_holdout | 0.887 | 0.931 | 3630.5 | 151.2 |
| linear_softmax_distilled_mlp_teacher_0p25 | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_distilled_mlp_teacher_0p25 | variant_logic_holdout | 0.898 | 0.932 | 3889.2 | 103.0 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher_0p25 | reasoning | 0.893 | 0.932 | 3759.9 |
| linear_softmax_distilled_mlp_teacher_0p25 | retrieval | 1.000 | 1.000 | 2763.6 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_distilled_mlp_teacher_0p25 | arithmetic | 0.875 | 0.931 | 3314.4 |
| linear_softmax_distilled_mlp_teacher_0p25 | logic | 0.898 | 0.932 | 3889.2 |
| linear_softmax_distilled_mlp_teacher_0p25 | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_distilled_mlp_teacher_0p25 | transcript | 1.000 | 1.000 | 3027.6 |
