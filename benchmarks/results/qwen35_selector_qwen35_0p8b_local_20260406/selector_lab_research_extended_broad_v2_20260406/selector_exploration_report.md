# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| static_rule | ok | row_multiclass | research_extended | global | yes | no | 0.750 | 0.736 | 3141.2 | 364.6 |
| linear_softmax | ok | row_multiclass | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_weighted | ok | row_multiclass | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_safe_router_global | ok | candidate_safe | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_safe_router_by_family | ok | candidate_safe | research_extended | per_prompt_family | yes | no | 0.983 | 0.983 | 3457.1 | 0.0 |
| candidate_target_linear_global | ok | candidate_target | research_extended | global | no | no | 0.759 | 0.500 | 4703.5 | 1885.7 |
| candidate_target_linear_by_family | ok | candidate_target | research_extended | per_prompt_family | no | no | 0.828 | 0.500 | 4835.5 | 1877.6 |
| candidate_target_mlp_global | ok | candidate_target | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_mlp_by_family | ok | candidate_target | research_extended | per_prompt_family | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_gbdt_global | ok | candidate_target | research_extended | global | no | no | 0.889 | 0.607 | 3802.7 | 428.4 |
| candidate_target_gbdt_by_family | ok | candidate_target | research_extended | per_prompt_family | no | no | 0.889 | 0.714 | 3601.6 | 227.3 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| static_rule | family_reasoning_holdout | 0.744 | 0.863 | 3639.3 | 561.2 |
| static_rule | retrieval_layer23_holdout | 0.750 | 0.750 | 1959.0 | 0.0 |
| static_rule | variant_logic_holdout | 0.727 | 0.841 | 3825.2 | 532.8 |
| linear_softmax | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| linear_softmax_compression_weighted | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_safe_router_global | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_safe_router_global | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_safe_router_global | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_safe_router_by_family | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_safe_router_by_family | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_safe_router_by_family | variant_logic_holdout | 0.966 | 0.966 | 3889.2 | 0.0 |
| candidate_target_linear_global | family_reasoning_holdout | 0.537 | 0.756 | 4607.3 | 1629.1 |
| candidate_target_linear_global | retrieval_layer23_holdout | 0.500 | 0.857 | 4573.9 | 2346.7 |
| candidate_target_linear_global | variant_logic_holdout | 0.534 | 0.761 | 4929.2 | 1681.2 |
| candidate_target_linear_by_family | family_reasoning_holdout | 0.531 | 0.894 | 5003.3 | 1604.9 |
| candidate_target_linear_by_family | retrieval_layer23_holdout | 0.500 | 0.857 | 4573.9 | 2346.7 |
| candidate_target_linear_by_family | variant_logic_holdout | 0.534 | 0.761 | 4929.2 | 1681.2 |
| candidate_target_mlp_global | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_mlp_global | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_mlp_global | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_mlp_by_family | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_mlp_by_family | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_mlp_by_family | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_gbdt_global | family_reasoning_holdout | 0.838 | 0.881 | 3797.7 | 279.6 |
| candidate_target_gbdt_global | retrieval_layer23_holdout | 0.607 | 1.000 | 3769.3 | 1005.7 |
| candidate_target_gbdt_global | variant_logic_holdout | 0.898 | 0.898 | 3841.2 | 0.0 |
| candidate_target_gbdt_by_family | family_reasoning_holdout | 0.838 | 0.881 | 3797.7 | 279.6 |
| candidate_target_gbdt_by_family | retrieval_layer23_holdout | 0.714 | 1.000 | 3165.9 | 402.3 |
| candidate_target_gbdt_by_family | variant_logic_holdout | 0.898 | 0.898 | 3841.2 | 0.0 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| static_rule | reasoning | 0.736 | 0.852 | 3732.3 |
| static_rule | retrieval | 0.750 | 0.750 | 1959.0 |
| linear_softmax | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_weighted | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_safe_router_global | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_safe_router_global | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_safe_router_by_family | reasoning | 0.983 | 0.983 | 3803.9 |
| candidate_safe_router_by_family | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_linear_global | reasoning | 0.536 | 0.759 | 4768.3 |
| candidate_target_linear_global | retrieval | 0.500 | 0.857 | 4573.9 |
| candidate_target_linear_by_family | reasoning | 0.533 | 0.828 | 4966.3 |
| candidate_target_linear_by_family | retrieval | 0.500 | 0.857 | 4573.9 |
| candidate_target_mlp_global | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_mlp_global | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_mlp_by_family | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_mlp_by_family | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_gbdt_global | reasoning | 0.868 | 0.889 | 3819.5 |
| candidate_target_gbdt_global | retrieval | 0.607 | 1.000 | 3769.3 |
| candidate_target_gbdt_by_family | reasoning | 0.868 | 0.889 | 3819.5 |
| candidate_target_gbdt_by_family | retrieval | 0.714 | 1.000 | 3165.9 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| static_rule | arithmetic | 0.764 | 0.889 | 3412.2 |
| static_rule | logic | 0.727 | 0.841 | 3825.2 |
| static_rule | memo | 0.750 | 0.750 | 1707.6 |
| static_rule | transcript | 0.750 | 0.750 | 2147.6 |
| linear_softmax | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_weighted | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_safe_router_global | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_safe_router_global | logic | 1.000 | 1.000 | 3985.2 |
| candidate_safe_router_global | memo | 1.000 | 1.000 | 2411.6 |
| candidate_safe_router_global | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_safe_router_by_family | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_safe_router_by_family | logic | 0.983 | 0.983 | 3937.2 |
| candidate_safe_router_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_safe_router_by_family | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_linear_global | arithmetic | 0.542 | 0.750 | 4213.9 |
| candidate_target_linear_global | logic | 0.534 | 0.761 | 4929.2 |
| candidate_target_linear_global | memo | 0.583 | 0.833 | 3584.9 |
| candidate_target_linear_global | transcript | 0.438 | 0.875 | 5315.6 |
| candidate_target_linear_by_family | arithmetic | 0.500 | 0.889 | 4663.7 |
| candidate_target_linear_by_family | logic | 0.545 | 0.830 | 5105.2 |
| candidate_target_linear_by_family | memo | 0.583 | 0.833 | 3584.9 |
| candidate_target_linear_by_family | transcript | 0.438 | 0.875 | 5315.6 |
| candidate_target_mlp_global | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_mlp_global | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_mlp_global | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_mlp_global | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_mlp_by_family | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_mlp_by_family | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_mlp_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_mlp_by_family | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_gbdt_global | arithmetic | 0.819 | 0.861 | 3431.7 |
| candidate_target_gbdt_global | logic | 0.875 | 0.898 | 3969.2 |
| candidate_target_gbdt_global | memo | 0.583 | 1.000 | 3350.2 |
| candidate_target_gbdt_global | transcript | 0.625 | 1.000 | 4083.6 |
| candidate_target_gbdt_by_family | arithmetic | 0.819 | 0.861 | 3431.7 |
| candidate_target_gbdt_by_family | logic | 0.875 | 0.898 | 3969.2 |
| candidate_target_gbdt_by_family | memo | 0.667 | 1.000 | 2880.9 |
| candidate_target_gbdt_by_family | transcript | 0.750 | 1.000 | 3379.6 |
