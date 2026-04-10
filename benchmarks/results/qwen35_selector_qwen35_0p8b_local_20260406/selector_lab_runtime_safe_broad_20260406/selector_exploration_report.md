# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| static_rule | ok | row_multiclass | runtime_safe | global | no | no | 0.750 | 0.736 | 3141.2 | 364.6 |
| linear_softmax | ok | row_multiclass | runtime_safe | global | no | no | 0.943 | 0.905 | 3506.0 | 150.7 |
| linear_softmax_compression_weighted | ok | row_multiclass | runtime_safe | global | yes | yes | 1.000 | 1.000 | 3489.1 | 0.0 |
| linear_softmax_compression_calibrated | ok | row_multiclass | runtime_safe | global | yes | yes | 0.771 | 0.762 | 2853.8 | 42.9 |
| candidate_safe_router_global | ok | candidate_safe | runtime_safe | global | yes | yes | 0.649 | 0.644 | 2747.9 | 33.5 |
| candidate_safe_router_by_family | ok | candidate_safe | runtime_safe | per_prompt_family | yes | yes | 0.799 | 0.722 | 3246.5 | 240.8 |
| candidate_target_linear_global | ok | candidate_target | runtime_safe | global | no | no | 0.759 | 0.500 | 4703.5 | 1885.7 |
| candidate_target_linear_by_family | ok | candidate_target | runtime_safe | per_prompt_family | no | no | 0.828 | 0.500 | 4832.5 | 1874.3 |
| candidate_target_mlp_global | ok | candidate_target | runtime_safe | global | no | no | 0.962 | 0.920 | 3552.4 | 161.1 |
| candidate_target_mlp_by_family | ok | candidate_target | runtime_safe | per_prompt_family | no | no | 0.962 | 0.920 | 3552.4 | 161.1 |
| candidate_target_gbdt_global | ok | candidate_target | runtime_safe | global | no | no | 0.982 | 0.672 | 4777.3 | 1321.2 |
| candidate_target_gbdt_by_family | ok | candidate_target | runtime_safe | per_prompt_family | no | no | 0.869 | 0.621 | 4448.9 | 1211.5 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| static_rule | family_reasoning_holdout | 0.744 | 0.863 | 3639.3 | 561.2 |
| static_rule | retrieval_layer23_holdout | 0.750 | 0.750 | 1959.0 | 0.0 |
| static_rule | variant_logic_holdout | 0.727 | 0.841 | 3825.2 | 532.8 |
| linear_softmax | family_reasoning_holdout | 0.900 | 0.944 | 3648.1 | 149.2 |
| linear_softmax | retrieval_layer23_holdout | 0.964 | 1.000 | 2964.7 | 201.1 |
| linear_softmax | variant_logic_holdout | 0.909 | 0.943 | 3905.2 | 101.8 |
| linear_softmax_compression_weighted | family_reasoning_holdout | 1.000 | 1.000 | 3718.5 | 0.0 |
| linear_softmax_compression_weighted | retrieval_layer23_holdout | 1.000 | 1.000 | 2763.6 | 0.0 |
| linear_softmax_compression_weighted | variant_logic_holdout | 1.000 | 1.000 | 3985.2 | 0.0 |
| linear_softmax_compression_calibrated | family_reasoning_holdout | 0.762 | 0.769 | 3005.7 | 45.8 |
| linear_softmax_compression_calibrated | retrieval_layer23_holdout | 0.821 | 0.821 | 2210.4 | 0.0 |
| linear_softmax_compression_calibrated | variant_logic_holdout | 0.761 | 0.773 | 3345.2 | 82.8 |
| candidate_safe_router_global | family_reasoning_holdout | 0.662 | 0.662 | 2715.3 | 0.0 |
| candidate_safe_router_global | retrieval_layer23_holdout | 0.929 | 0.929 | 2663.0 | 0.0 |
| candidate_safe_router_global | variant_logic_holdout | 0.625 | 0.636 | 2865.2 | 100.6 |
| candidate_safe_router_by_family | family_reasoning_holdout | 0.819 | 0.963 | 4211.3 | 621.7 |
| candidate_safe_router_by_family | retrieval_layer23_holdout | 0.929 | 0.929 | 2663.0 | 0.0 |
| candidate_safe_router_by_family | variant_logic_holdout | 0.625 | 0.636 | 2865.2 | 100.6 |
| candidate_target_linear_global | family_reasoning_holdout | 0.537 | 0.756 | 4607.3 | 1629.1 |
| candidate_target_linear_global | retrieval_layer23_holdout | 0.500 | 0.857 | 4573.9 | 2346.7 |
| candidate_target_linear_global | variant_logic_holdout | 0.534 | 0.761 | 4929.2 | 1681.2 |
| candidate_target_linear_by_family | family_reasoning_holdout | 0.537 | 0.894 | 4994.5 | 1595.1 |
| candidate_target_linear_by_family | retrieval_layer23_holdout | 0.500 | 0.857 | 4573.9 | 2346.7 |
| candidate_target_linear_by_family | variant_logic_holdout | 0.534 | 0.761 | 4929.2 | 1681.2 |
| candidate_target_mlp_global | family_reasoning_holdout | 0.919 | 0.969 | 3771.3 | 181.7 |
| candidate_target_mlp_global | retrieval_layer23_holdout | 0.964 | 1.000 | 2964.7 | 201.1 |
| candidate_target_mlp_global | variant_logic_holdout | 0.920 | 0.955 | 3921.2 | 100.6 |
| candidate_target_mlp_by_family | family_reasoning_holdout | 0.919 | 0.969 | 3771.3 | 181.7 |
| candidate_target_mlp_by_family | retrieval_layer23_holdout | 0.964 | 1.000 | 2964.7 | 201.1 |
| candidate_target_mlp_by_family | variant_logic_holdout | 0.920 | 0.955 | 3921.2 | 100.6 |
| candidate_target_gbdt_global | family_reasoning_holdout | 0.662 | 0.988 | 5082.5 | 1399.1 |
| candidate_target_gbdt_global | retrieval_layer23_holdout | 0.714 | 1.000 | 3920.1 | 1156.6 |
| candidate_target_gbdt_global | variant_logic_holdout | 0.682 | 0.977 | 5329.2 | 1408.0 |
| candidate_target_gbdt_by_family | family_reasoning_holdout | 0.662 | 0.988 | 5082.5 | 1399.1 |
| candidate_target_gbdt_by_family | retrieval_layer23_holdout | 0.750 | 1.000 | 3719.0 | 955.4 |
| candidate_target_gbdt_by_family | variant_logic_holdout | 0.580 | 0.750 | 4545.2 | 1280.0 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| static_rule | reasoning | 0.736 | 0.852 | 3732.3 |
| static_rule | retrieval | 0.750 | 0.750 | 1959.0 |
| linear_softmax | reasoning | 0.905 | 0.943 | 3776.7 |
| linear_softmax | retrieval | 0.964 | 1.000 | 2964.7 |
| linear_softmax_compression_weighted | reasoning | 1.000 | 1.000 | 3851.9 |
| linear_softmax_compression_weighted | retrieval | 1.000 | 1.000 | 2763.6 |
| linear_softmax_compression_calibrated | reasoning | 0.762 | 0.771 | 3175.5 |
| linear_softmax_compression_calibrated | retrieval | 0.821 | 0.821 | 2210.4 |
| candidate_safe_router_global | reasoning | 0.644 | 0.649 | 2790.3 |
| candidate_safe_router_global | retrieval | 0.929 | 0.929 | 2663.0 |
| candidate_safe_router_by_family | reasoning | 0.722 | 0.799 | 3538.3 |
| candidate_safe_router_by_family | retrieval | 0.929 | 0.929 | 2663.0 |
| candidate_target_linear_global | reasoning | 0.536 | 0.759 | 4768.3 |
| candidate_target_linear_global | retrieval | 0.500 | 0.857 | 4573.9 |
| candidate_target_linear_by_family | reasoning | 0.536 | 0.828 | 4961.9 |
| candidate_target_linear_by_family | retrieval | 0.500 | 0.857 | 4573.9 |
| candidate_target_mlp_global | reasoning | 0.920 | 0.962 | 3846.3 |
| candidate_target_mlp_global | retrieval | 0.964 | 1.000 | 2964.7 |
| candidate_target_mlp_by_family | reasoning | 0.920 | 0.962 | 3846.3 |
| candidate_target_mlp_by_family | retrieval | 0.964 | 1.000 | 2964.7 |
| candidate_target_gbdt_global | reasoning | 0.672 | 0.982 | 5205.9 |
| candidate_target_gbdt_global | retrieval | 0.714 | 1.000 | 3920.1 |
| candidate_target_gbdt_by_family | reasoning | 0.621 | 0.869 | 4813.9 |
| candidate_target_gbdt_by_family | retrieval | 0.750 | 1.000 | 3719.0 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| static_rule | arithmetic | 0.764 | 0.889 | 3412.2 |
| static_rule | logic | 0.727 | 0.841 | 3825.2 |
| static_rule | memo | 0.750 | 0.750 | 1707.6 |
| static_rule | transcript | 0.750 | 0.750 | 2147.6 |
| linear_softmax | arithmetic | 0.889 | 0.944 | 3333.9 |
| linear_softmax | logic | 0.909 | 0.943 | 3905.2 |
| linear_softmax | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax | transcript | 0.938 | 1.000 | 3379.6 |
| linear_softmax_compression_weighted | arithmetic | 1.000 | 1.000 | 3392.6 |
| linear_softmax_compression_weighted | logic | 1.000 | 1.000 | 3985.2 |
| linear_softmax_compression_weighted | memo | 1.000 | 1.000 | 2411.6 |
| linear_softmax_compression_weighted | transcript | 1.000 | 1.000 | 3027.6 |
| linear_softmax_compression_calibrated | arithmetic | 0.750 | 0.750 | 2571.3 |
| linear_softmax_compression_calibrated | logic | 0.767 | 0.778 | 3353.2 |
| linear_softmax_compression_calibrated | memo | 0.750 | 0.750 | 1707.6 |
| linear_softmax_compression_calibrated | transcript | 0.875 | 0.875 | 2587.6 |
| candidate_safe_router_global | arithmetic | 0.653 | 0.653 | 2375.7 |
| candidate_safe_router_global | logic | 0.648 | 0.653 | 2929.2 |
| candidate_safe_router_global | memo | 0.917 | 0.917 | 2294.2 |
| candidate_safe_router_global | transcript | 0.938 | 0.938 | 2939.6 |
| candidate_safe_router_by_family | arithmetic | 0.778 | 0.972 | 4096.6 |
| candidate_safe_router_by_family | logic | 0.739 | 0.795 | 3585.2 |
| candidate_safe_router_by_family | memo | 0.917 | 0.917 | 2294.2 |
| candidate_safe_router_by_family | transcript | 0.938 | 0.938 | 2939.6 |
| candidate_target_linear_global | arithmetic | 0.542 | 0.750 | 4213.9 |
| candidate_target_linear_global | logic | 0.534 | 0.761 | 4929.2 |
| candidate_target_linear_global | memo | 0.583 | 0.833 | 3584.9 |
| candidate_target_linear_global | transcript | 0.438 | 0.875 | 5315.6 |
| candidate_target_linear_by_family | arithmetic | 0.514 | 0.889 | 4644.2 |
| candidate_target_linear_by_family | logic | 0.545 | 0.830 | 5105.2 |
| candidate_target_linear_by_family | memo | 0.583 | 0.833 | 3584.9 |
| candidate_target_linear_by_family | transcript | 0.438 | 0.875 | 5315.6 |
| candidate_target_mlp_global | arithmetic | 0.917 | 0.986 | 3588.2 |
| candidate_target_mlp_global | logic | 0.920 | 0.955 | 3921.2 |
| candidate_target_mlp_global | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_mlp_global | transcript | 0.938 | 1.000 | 3379.6 |
| candidate_target_mlp_by_family | arithmetic | 0.917 | 0.986 | 3588.2 |
| candidate_target_mlp_by_family | logic | 0.920 | 0.955 | 3921.2 |
| candidate_target_mlp_by_family | memo | 1.000 | 1.000 | 2411.6 |
| candidate_target_mlp_by_family | transcript | 0.938 | 1.000 | 3379.6 |
| candidate_target_gbdt_global | arithmetic | 0.681 | 1.000 | 4722.4 |
| candidate_target_gbdt_global | logic | 0.665 | 0.977 | 5353.2 |
| candidate_target_gbdt_global | memo | 0.583 | 1.000 | 4054.2 |
| candidate_target_gbdt_global | transcript | 0.812 | 1.000 | 3819.6 |
| candidate_target_gbdt_by_family | arithmetic | 0.681 | 1.000 | 4722.4 |
| candidate_target_gbdt_by_family | logic | 0.614 | 0.864 | 4961.2 |
| candidate_target_gbdt_by_family | memo | 0.583 | 1.000 | 4054.2 |
| candidate_target_gbdt_by_family | transcript | 0.875 | 1.000 | 3467.6 |
