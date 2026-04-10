# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_floor_0p95 | ok | row_multiclass | runtime_safe | global | yes | no | 0.771 | 0.762 | 2853.8 | 42.9 |
| linear_softmax_compression_floor_0p90 | ok | row_multiclass | runtime_safe | global | no | no | 0.771 | 0.762 | 3038.2 | 42.9 |
| linear_softmax_compression_floor_0p85 | ok | row_multiclass | runtime_safe | global | yes | no | 0.940 | 0.901 | 3419.3 | 84.0 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| linear_softmax_compression_floor_0p95 | family_reasoning_holdout | 0.762 | 0.769 | 3005.7 | 45.8 |
| linear_softmax_compression_floor_0p95 | retrieval_layer23_holdout | 0.821 | 0.821 | 2210.4 | 0.0 |
| linear_softmax_compression_floor_0p95 | variant_logic_holdout | 0.761 | 0.773 | 3345.2 | 82.8 |
| linear_softmax_compression_floor_0p90 | family_reasoning_holdout | 0.762 | 0.769 | 3005.7 | 45.8 |
| linear_softmax_compression_floor_0p90 | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_floor_0p90 | variant_logic_holdout | 0.761 | 0.773 | 3345.2 | 82.8 |
| linear_softmax_compression_floor_0p85 | family_reasoning_holdout | 0.894 | 0.938 | 3639.3 | 150.2 |
| linear_softmax_compression_floor_0p85 | retrieval_layer23_holdout | 0.964 | 0.964 | 2713.3 | 0.0 |
| linear_softmax_compression_floor_0p85 | variant_logic_holdout | 0.909 | 0.943 | 3905.2 | 101.8 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_floor_0p95 | reasoning | 0.762 | 0.771 | 3175.5 |
| linear_softmax_compression_floor_0p95 | retrieval | 0.821 | 0.821 | 2210.4 |
| linear_softmax_compression_floor_0p90 | reasoning | 0.762 | 0.771 | 3175.5 |
| linear_softmax_compression_floor_0p90 | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_floor_0p85 | reasoning | 0.901 | 0.940 | 3772.3 |
| linear_softmax_compression_floor_0p85 | retrieval | 0.964 | 0.964 | 2713.3 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_floor_0p95 | arithmetic | 0.750 | 0.750 | 2571.3 |
| linear_softmax_compression_floor_0p95 | logic | 0.767 | 0.778 | 3353.2 |
| linear_softmax_compression_floor_0p95 | memo | 0.750 | 0.750 | 1707.6 |
| linear_softmax_compression_floor_0p95 | transcript | 0.875 | 0.875 | 2587.6 |
| linear_softmax_compression_floor_0p90 | arithmetic | 0.750 | 0.750 | 2571.3 |
| linear_softmax_compression_floor_0p90 | logic | 0.767 | 0.778 | 3353.2 |
| linear_softmax_compression_floor_0p90 | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_floor_0p90 | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_floor_0p85 | arithmetic | 0.875 | 0.931 | 3314.4 |
| linear_softmax_compression_floor_0p85 | logic | 0.909 | 0.943 | 3905.2 |
| linear_softmax_compression_floor_0p85 | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_floor_0p85 | transcript | 0.938 | 0.938 | 2939.6 |
