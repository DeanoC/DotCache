# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_linear_by_family | ok | candidate_target | runtime_safe | per_prompt_family | no | no | 0.828 | 0.500 | 4832.5 | 1874.3 |
| candidate_target_trace_softmax_by_family | ok | candidate_target | runtime_safe | per_prompt_family | no | no | 0.759 | 0.500 | 4703.5 | 1885.7 |
| candidate_target_dense_mix_light_by_family | ok | candidate_target | runtime_safe | per_prompt_family | yes | no | 0.492 | 0.492 | 1991.9 | 0.0 |
| candidate_target_dense_mix_medium_by_family | ok | candidate_target | runtime_safe | per_prompt_family | yes | no | 0.492 | 0.492 | 1991.9 | 0.0 |
| candidate_target_dense_mix_heavy_by_family | ok | candidate_target | runtime_safe | per_prompt_family | yes | no | 0.492 | 0.492 | 1991.9 | 0.0 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_linear_by_family | family_reasoning_holdout | 0.537 | 0.894 | 4994.5 | 1595.1 |
| candidate_target_linear_by_family | retrieval_layer23_holdout | 0.500 | 0.857 | 4573.9 | 2346.7 |
| candidate_target_linear_by_family | variant_logic_holdout | 0.534 | 0.761 | 4929.2 | 1681.2 |
| candidate_target_trace_softmax_by_family | family_reasoning_holdout | 0.537 | 0.756 | 4607.3 | 1629.1 |
| candidate_target_trace_softmax_by_family | retrieval_layer23_holdout | 0.500 | 0.857 | 4573.9 | 2346.7 |
| candidate_target_trace_softmax_by_family | variant_logic_holdout | 0.534 | 0.761 | 4929.2 | 1681.2 |
| candidate_target_dense_mix_light_by_family | family_reasoning_holdout | 0.506 | 0.506 | 1967.3 | 0.0 |
| candidate_target_dense_mix_light_by_family | retrieval_layer23_holdout | 0.750 | 0.750 | 1959.0 | 0.0 |
| candidate_target_dense_mix_light_by_family | variant_logic_holdout | 0.477 | 0.477 | 2049.2 | 0.0 |
| candidate_target_dense_mix_medium_by_family | family_reasoning_holdout | 0.506 | 0.506 | 1967.3 | 0.0 |
| candidate_target_dense_mix_medium_by_family | retrieval_layer23_holdout | 0.750 | 0.750 | 1959.0 | 0.0 |
| candidate_target_dense_mix_medium_by_family | variant_logic_holdout | 0.477 | 0.477 | 2049.2 | 0.0 |
| candidate_target_dense_mix_heavy_by_family | family_reasoning_holdout | 0.506 | 0.506 | 1967.3 | 0.0 |
| candidate_target_dense_mix_heavy_by_family | retrieval_layer23_holdout | 0.750 | 0.750 | 1959.0 | 0.0 |
| candidate_target_dense_mix_heavy_by_family | variant_logic_holdout | 0.477 | 0.477 | 2049.2 | 0.0 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_linear_by_family | reasoning | 0.536 | 0.828 | 4961.9 |
| candidate_target_linear_by_family | retrieval | 0.500 | 0.857 | 4573.9 |
| candidate_target_trace_softmax_by_family | reasoning | 0.536 | 0.759 | 4768.3 |
| candidate_target_trace_softmax_by_family | retrieval | 0.500 | 0.857 | 4573.9 |
| candidate_target_dense_mix_light_by_family | reasoning | 0.492 | 0.492 | 2008.3 |
| candidate_target_dense_mix_light_by_family | retrieval | 0.750 | 0.750 | 1959.0 |
| candidate_target_dense_mix_medium_by_family | reasoning | 0.492 | 0.492 | 2008.3 |
| candidate_target_dense_mix_medium_by_family | retrieval | 0.750 | 0.750 | 1959.0 |
| candidate_target_dense_mix_heavy_by_family | reasoning | 0.492 | 0.492 | 2008.3 |
| candidate_target_dense_mix_heavy_by_family | retrieval | 0.750 | 0.750 | 1959.0 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_linear_by_family | arithmetic | 0.514 | 0.889 | 4644.2 |
| candidate_target_linear_by_family | logic | 0.545 | 0.830 | 5105.2 |
| candidate_target_linear_by_family | memo | 0.583 | 0.833 | 3584.9 |
| candidate_target_linear_by_family | transcript | 0.438 | 0.875 | 5315.6 |
| candidate_target_trace_softmax_by_family | arithmetic | 0.542 | 0.750 | 4213.9 |
| candidate_target_trace_softmax_by_family | logic | 0.534 | 0.761 | 4929.2 |
| candidate_target_trace_softmax_by_family | memo | 0.583 | 0.833 | 3584.9 |
| candidate_target_trace_softmax_by_family | transcript | 0.438 | 0.875 | 5315.6 |
| candidate_target_dense_mix_light_by_family | arithmetic | 0.542 | 0.542 | 1867.3 |
| candidate_target_dense_mix_light_by_family | logic | 0.477 | 0.477 | 2049.2 |
| candidate_target_dense_mix_light_by_family | memo | 0.750 | 0.750 | 1707.6 |
| candidate_target_dense_mix_light_by_family | transcript | 0.750 | 0.750 | 2147.6 |
| candidate_target_dense_mix_medium_by_family | arithmetic | 0.542 | 0.542 | 1867.3 |
| candidate_target_dense_mix_medium_by_family | logic | 0.477 | 0.477 | 2049.2 |
| candidate_target_dense_mix_medium_by_family | memo | 0.750 | 0.750 | 1707.6 |
| candidate_target_dense_mix_medium_by_family | transcript | 0.750 | 0.750 | 2147.6 |
| candidate_target_dense_mix_heavy_by_family | arithmetic | 0.542 | 0.542 | 1867.3 |
| candidate_target_dense_mix_heavy_by_family | logic | 0.477 | 0.477 | 2049.2 |
| candidate_target_dense_mix_heavy_by_family | memo | 0.750 | 0.750 | 1707.6 |
| candidate_target_dense_mix_heavy_by_family | transcript | 0.750 | 0.750 | 2147.6 |
