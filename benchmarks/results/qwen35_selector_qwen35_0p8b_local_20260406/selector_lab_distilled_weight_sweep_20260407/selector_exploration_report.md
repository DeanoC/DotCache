# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| distilled_teacher_weight_0p25 | ok | row_multiclass | runtime_safe | global | yes | no | 0.932 | 0.893 | 3427.8 | 84.7 |
| distilled_teacher_weight_0p50 | ok | row_multiclass | runtime_safe | global | yes | no | 0.928 | 0.889 | 3424.8 | 85.1 |
| distilled_teacher_weight_0p75 | ok | row_multiclass | runtime_safe | global | yes | no | 0.923 | 0.887 | 3416.6 | 82.3 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| distilled_teacher_weight_0p25 | family_reasoning_holdout | 0.887 | 0.931 | 3630.5 | 151.2 |
| distilled_teacher_weight_0p25 | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| distilled_teacher_weight_0p25 | variant_logic_holdout | 0.898 | 0.932 | 3889.2 | 103.0 |
| distilled_teacher_weight_0p50 | family_reasoning_holdout | 0.881 | 0.925 | 3621.7 | 152.2 |
| distilled_teacher_weight_0p50 | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| distilled_teacher_weight_0p50 | variant_logic_holdout | 0.898 | 0.932 | 3889.2 | 103.0 |
| distilled_teacher_weight_0p75 | family_reasoning_holdout | 0.887 | 0.925 | 3612.9 | 142.7 |
| distilled_teacher_weight_0p75 | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| distilled_teacher_weight_0p75 | variant_logic_holdout | 0.886 | 0.920 | 3873.2 | 104.3 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted | retrieval | 1.000 | 1.000 | 2763.6 |
| distilled_teacher_weight_0p25 | reasoning | 0.893 | 0.932 | 3759.9 |
| distilled_teacher_weight_0p25 | retrieval | 1.000 | 1.000 | 2763.6 |
| distilled_teacher_weight_0p50 | reasoning | 0.889 | 0.928 | 3755.5 |
| distilled_teacher_weight_0p50 | retrieval | 1.000 | 1.000 | 2763.6 |
| distilled_teacher_weight_0p75 | reasoning | 0.887 | 0.923 | 3743.1 |
| distilled_teacher_weight_0p75 | retrieval | 1.000 | 1.000 | 2763.6 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted | transcript | 1.000 | 1.000 | 3027.6 |
| distilled_teacher_weight_0p25 | arithmetic | 0.875 | 0.931 | 3314.4 |
| distilled_teacher_weight_0p25 | logic | 0.898 | 0.932 | 3889.2 |
| distilled_teacher_weight_0p25 | memo | 1.000 | 1.000 | 2411.6 |
| distilled_teacher_weight_0p25 | transcript | 1.000 | 1.000 | 3027.6 |
| distilled_teacher_weight_0p50 | arithmetic | 0.875 | 0.931 | 3314.4 |
| distilled_teacher_weight_0p50 | logic | 0.892 | 0.926 | 3881.2 |
| distilled_teacher_weight_0p50 | memo | 1.000 | 1.000 | 2411.6 |
| distilled_teacher_weight_0p50 | transcript | 1.000 | 1.000 | 3027.6 |
| distilled_teacher_weight_0p75 | arithmetic | 0.889 | 0.931 | 3294.8 |
| distilled_teacher_weight_0p75 | logic | 0.886 | 0.920 | 3873.2 |
| distilled_teacher_weight_0p75 | memo | 1.000 | 1.000 | 2411.6 |
| distilled_teacher_weight_0p75 | transcript | 1.000 | 1.000 | 3027.6 |
