# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| static_rule | ok | row_multiclass | research_extended | global | yes | no | 0.750 | 0.736 | 3141.2 | 364.6 |
| linear_softmax | ok | row_multiclass | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_weighted | ok | row_multiclass | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_safe_router_global | ok | candidate_safe | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_safe_router_by_family | ok | candidate_safe | research_extended | per_prompt_family | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_linear_global | ok | candidate_target | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_linear_by_family | ok | candidate_target | research_extended | per_prompt_family | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_mlp_global | ok | candidate_target | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_mlp_by_family | ok | candidate_target | research_extended | per_prompt_family | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_gbdt_global | ok | candidate_target | research_extended | global | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |
| candidate_target_gbdt_by_family | ok | candidate_target | research_extended | per_prompt_family | yes | no | 1.000 | 1.000 | 3489.1 | 0.0 |

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
| candidate_safe_router_by_family | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_linear_global | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_linear_global | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_linear_global | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_linear_by_family | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_linear_by_family | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_linear_by_family | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_mlp_global | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_mlp_global | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_mlp_global | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_mlp_by_family | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_mlp_by_family | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_mlp_by_family | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_gbdt_global | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_gbdt_global | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_gbdt_global | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| candidate_target_gbdt_by_family | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| candidate_target_gbdt_by_family | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| candidate_target_gbdt_by_family | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |

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
| candidate_safe_router_by_family | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_safe_router_by_family | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_linear_global | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_linear_global | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_linear_by_family | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_linear_by_family | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_mlp_global | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_mlp_global | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_mlp_by_family | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_mlp_by_family | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_gbdt_global | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_gbdt_global | retrieval | 1.000 | 1.000 | 2763.6 |
| candidate_target_gbdt_by_family | reasoning | 1.000 | 1.000 | 3851.9 |
| candidate_target_gbdt_by_family | retrieval | 1.000 | 1.000 | 2763.6 |

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
| candidate_safe_router_by_family | logic | 1.000 | 1.000 | 3985.2 |
| candidate_safe_router_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_safe_router_by_family | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_linear_global | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_linear_global | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_linear_global | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_linear_global | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_linear_by_family | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_linear_by_family | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_linear_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_linear_by_family | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_mlp_global | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_mlp_global | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_mlp_global | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_mlp_global | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_mlp_by_family | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_mlp_by_family | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_mlp_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_mlp_by_family | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_gbdt_global | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_gbdt_global | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_gbdt_global | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_gbdt_global | transcript | 1.000 | 1.000 | 3027.6 |
| candidate_target_gbdt_by_family | arithmetic | 1.000 | 1.000 | 3392.6 |
| candidate_target_gbdt_by_family | logic | 1.000 | 1.000 | 3985.2 |
| candidate_target_gbdt_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_gbdt_by_family | transcript | 1.000 | 1.000 | 3027.6 |
