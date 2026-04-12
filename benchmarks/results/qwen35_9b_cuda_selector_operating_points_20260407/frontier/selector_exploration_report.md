# Selector Exploration Lab

## Aggregate

| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted_dense_control | ok | row_multiclass | runtime_safe | global | yes | no | 1.000 | 1.000 | 4854.5 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | ok | row_multiclass | runtime_safe | global | yes | no | 0.627 | 0.618 | 3752.7 | 87.7 |
| linear_softmax_compression_floor_0p90_dense_control | ok | row_multiclass | runtime_safe | global | yes | no | 0.627 | 0.618 | 3752.7 | 87.7 |
| linear_softmax_compression_floor_0p85_dense_control | ok | row_multiclass | runtime_safe | global | no | no | 0.627 | 0.618 | 4060.9 | 179.8 |
| linear_softmax_compression_floor_0p80_dense_control | ok | row_multiclass | runtime_safe | global | yes | no | 0.738 | 0.625 | 4647.3 | 382.8 |
| linear_softmax_compression_floor_0p75_dense_control | ok | row_multiclass | runtime_safe | global | yes | no | 0.770 | 0.724 | 4673.7 | 322.0 |
| linear_softmax_compression_floor_0p70_dense_control | ok | row_multiclass | runtime_safe | global | yes | no | 0.668 | 0.655 | 4329.9 | 184.3 |

## By Split

| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |
| --- | --- | ---: | ---: | ---: | ---: |
| linear_softmax_compression_weighted_dense_control | cache_layer23_holdout | 1.000 | 1.000 | 5944.2 | 0.0 |
| linear_softmax_compression_weighted_dense_control | family_cache_holdout | 1.000 | 1.000 | 5805.3 | 0.0 |
| linear_softmax_compression_weighted_dense_control | family_instruction_holdout | 1.000 | 1.000 | 4849.0 | 0.0 |
| linear_softmax_compression_weighted_dense_control | family_reasoning_holdout | 1.000 | 1.000 | 4917.3 | 0.0 |
| linear_softmax_compression_weighted_dense_control | family_retrieval_holdout | 1.000 | 1.000 | 4502.7 | 0.0 |
| linear_softmax_compression_weighted_dense_control | reasoning_layer23_holdout | 1.000 | 1.000 | 5650.0 | 0.0 |
| linear_softmax_compression_weighted_dense_control | retrieval_layer23_holdout | 1.000 | 1.000 | 3186.8 | 0.0 |
| linear_softmax_compression_weighted_dense_control | variant_arithmetic_holdout | 1.000 | 1.000 | 5026.1 | 0.0 |
| linear_softmax_compression_weighted_dense_control | variant_constraints_holdout | 1.000 | 1.000 | 4781.8 | 0.0 |
| linear_softmax_compression_weighted_dense_control | variant_logic_holdout | 1.000 | 1.000 | 4810.0 | 0.0 |
| linear_softmax_compression_weighted_dense_control | variant_memo_holdout | 1.000 | 1.000 | 3925.8 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | cache_layer23_holdout | 0.850 | 0.850 | 5733.0 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | family_cache_holdout | 0.603 | 0.627 | 4442.0 | 213.9 |
| linear_softmax_compression_floor_0p95_dense_control | family_instruction_holdout | 0.596 | 0.614 | 3490.4 | 160.9 |
| linear_softmax_compression_floor_0p95_dense_control | family_reasoning_holdout | 0.589 | 0.589 | 3389.4 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | family_retrieval_holdout | 0.603 | 0.618 | 3137.7 | 104.3 |
| linear_softmax_compression_floor_0p95_dense_control | reasoning_layer23_holdout | 0.800 | 0.800 | 4992.9 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | retrieval_layer23_holdout | 0.778 | 0.889 | 2873.9 | 352.0 |
| linear_softmax_compression_floor_0p95_dense_control | variant_arithmetic_holdout | 0.586 | 0.586 | 3417.0 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | variant_constraints_holdout | 0.640 | 0.640 | 3458.3 | 0.0 |
| linear_softmax_compression_floor_0p95_dense_control | variant_logic_holdout | 0.577 | 0.592 | 3421.8 | 134.1 |
| linear_softmax_compression_floor_0p95_dense_control | variant_memo_holdout | 0.712 | 0.712 | 2923.2 | 0.0 |
| linear_softmax_compression_floor_0p90_dense_control | cache_layer23_holdout | 0.850 | 0.850 | 5733.0 | 0.0 |
| linear_softmax_compression_floor_0p90_dense_control | family_cache_holdout | 0.603 | 0.627 | 4442.0 | 213.9 |
| linear_softmax_compression_floor_0p90_dense_control | family_instruction_holdout | 0.596 | 0.614 | 3490.4 | 160.9 |
| linear_softmax_compression_floor_0p90_dense_control | family_reasoning_holdout | 0.589 | 0.589 | 3389.4 | 0.0 |
| linear_softmax_compression_floor_0p90_dense_control | family_retrieval_holdout | 0.603 | 0.618 | 3137.7 | 104.3 |
| linear_softmax_compression_floor_0p90_dense_control | reasoning_layer23_holdout | 0.800 | 0.800 | 4992.9 | 0.0 |
| linear_softmax_compression_floor_0p90_dense_control | retrieval_layer23_holdout | 0.778 | 0.889 | 2873.9 | 352.0 |
| linear_softmax_compression_floor_0p90_dense_control | variant_arithmetic_holdout | 0.586 | 0.586 | 3417.0 | 0.0 |
| linear_softmax_compression_floor_0p90_dense_control | variant_constraints_holdout | 0.640 | 0.640 | 3458.3 | 0.0 |
| linear_softmax_compression_floor_0p90_dense_control | variant_logic_holdout | 0.577 | 0.592 | 3421.8 | 134.1 |
| linear_softmax_compression_floor_0p90_dense_control | variant_memo_holdout | 0.712 | 0.712 | 2923.2 | 0.0 |
| linear_softmax_compression_floor_0p85_dense_control | cache_layer23_holdout | 0.850 | 0.850 | 5733.0 | 0.0 |
| linear_softmax_compression_floor_0p85_dense_control | family_cache_holdout | 0.603 | 0.627 | 4442.0 | 213.9 |
| linear_softmax_compression_floor_0p85_dense_control | family_instruction_holdout | 0.596 | 0.614 | 3490.4 | 160.9 |
| linear_softmax_compression_floor_0p85_dense_control | family_reasoning_holdout | 0.589 | 0.589 | 3389.4 | 0.0 |
| linear_softmax_compression_floor_0p85_dense_control | family_retrieval_holdout | 0.725 | 0.939 | 5179.9 | 892.9 |
| linear_softmax_compression_floor_0p85_dense_control | reasoning_layer23_holdout | 0.800 | 0.800 | 4992.9 | 0.0 |
| linear_softmax_compression_floor_0p85_dense_control | retrieval_layer23_holdout | 0.778 | 0.889 | 2873.9 | 352.0 |
| linear_softmax_compression_floor_0p85_dense_control | variant_arithmetic_holdout | 0.586 | 0.586 | 3417.0 | 0.0 |
| linear_softmax_compression_floor_0p85_dense_control | variant_constraints_holdout | 0.640 | 0.640 | 3458.3 | 0.0 |
| linear_softmax_compression_floor_0p85_dense_control | variant_logic_holdout | 0.831 | 0.887 | 4770.3 | 357.6 |
| linear_softmax_compression_floor_0p85_dense_control | variant_memo_holdout | 0.712 | 0.712 | 2923.2 | 0.0 |
| linear_softmax_compression_floor_0p80_dense_control | cache_layer23_holdout | 0.850 | 0.850 | 5733.0 | 0.0 |
| linear_softmax_compression_floor_0p80_dense_control | family_cache_holdout | 0.603 | 0.627 | 4442.0 | 213.9 |
| linear_softmax_compression_floor_0p80_dense_control | family_instruction_holdout | 0.781 | 0.904 | 4873.7 | 505.8 |
| linear_softmax_compression_floor_0p80_dense_control | family_reasoning_holdout | 0.766 | 0.965 | 5516.4 | 776.5 |
| linear_softmax_compression_floor_0p80_dense_control | family_retrieval_holdout | 0.718 | 0.901 | 4921.9 | 799.5 |
| linear_softmax_compression_floor_0p80_dense_control | reasoning_layer23_holdout | 0.800 | 0.800 | 4992.9 | 0.0 |
| linear_softmax_compression_floor_0p80_dense_control | retrieval_layer23_holdout | 0.444 | 1.000 | 4438.3 | 1251.6 |
| linear_softmax_compression_floor_0p80_dense_control | variant_arithmetic_holdout | 0.714 | 0.757 | 4261.8 | 239.1 |
| linear_softmax_compression_floor_0p80_dense_control | variant_constraints_holdout | 0.800 | 0.840 | 4246.8 | 67.0 |
| linear_softmax_compression_floor_0p80_dense_control | variant_logic_holdout | 0.831 | 0.887 | 4770.3 | 357.6 |
| linear_softmax_compression_floor_0p80_dense_control | variant_memo_holdout | 0.712 | 0.712 | 2923.2 | 0.0 |
| linear_softmax_compression_floor_0p75_dense_control | cache_layer23_holdout | 0.900 | 0.950 | 6155.5 | 296.4 |
| linear_softmax_compression_floor_0p75_dense_control | family_cache_holdout | 0.754 | 0.802 | 5257.8 | 306.7 |
| linear_softmax_compression_floor_0p75_dense_control | family_instruction_holdout | 0.737 | 0.781 | 4182.0 | 221.5 |
| linear_softmax_compression_floor_0p75_dense_control | family_reasoning_holdout | 0.816 | 0.901 | 4947.2 | 421.3 |
| linear_softmax_compression_floor_0p75_dense_control | family_retrieval_holdout | 0.748 | 0.847 | 4406.0 | 494.7 |
| linear_softmax_compression_floor_0p75_dense_control | reasoning_layer23_holdout | 0.933 | 0.933 | 5556.1 | 0.0 |
| linear_softmax_compression_floor_0p75_dense_control | retrieval_layer23_holdout | 0.667 | 1.000 | 3812.6 | 625.8 |
| linear_softmax_compression_floor_0p75_dense_control | variant_arithmetic_holdout | 0.714 | 0.757 | 4261.8 | 239.1 |
| linear_softmax_compression_floor_0p75_dense_control | variant_constraints_holdout | 0.720 | 0.760 | 3908.9 | 74.1 |
| linear_softmax_compression_floor_0p75_dense_control | variant_logic_holdout | 0.845 | 0.873 | 4591.8 | 181.7 |
| linear_softmax_compression_floor_0p75_dense_control | variant_memo_holdout | 0.758 | 0.909 | 4331.2 | 680.5 |
| linear_softmax_compression_floor_0p70_dense_control | cache_layer23_holdout | 0.900 | 0.900 | 5803.5 | 0.0 |
| linear_softmax_compression_floor_0p70_dense_control | family_cache_holdout | 0.754 | 0.802 | 5257.8 | 306.7 |
| linear_softmax_compression_floor_0p70_dense_control | family_instruction_holdout | 0.649 | 0.675 | 3675.6 | 164.6 |
| linear_softmax_compression_floor_0p70_dense_control | family_reasoning_holdout | 0.773 | 0.809 | 4418.0 | 210.0 |
| linear_softmax_compression_floor_0p70_dense_control | family_retrieval_holdout | 0.733 | 0.786 | 3965.3 | 328.1 |
| linear_softmax_compression_floor_0p70_dense_control | reasoning_layer23_holdout | 0.933 | 0.933 | 5556.1 | 0.0 |
| linear_softmax_compression_floor_0p70_dense_control | retrieval_layer23_holdout | 0.778 | 1.000 | 3656.1 | 469.3 |
| linear_softmax_compression_floor_0p70_dense_control | variant_arithmetic_holdout | 0.643 | 0.671 | 3899.7 | 149.8 |
| linear_softmax_compression_floor_0p70_dense_control | variant_constraints_holdout | 0.660 | 0.660 | 3486.5 | 0.0 |
| linear_softmax_compression_floor_0p70_dense_control | variant_logic_holdout | 0.761 | 0.775 | 4155.5 | 102.4 |
| linear_softmax_compression_floor_0p70_dense_control | variant_memo_holdout | 0.818 | 0.864 | 3755.2 | 296.4 |

## By Prompt Family

| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted_dense_control | cache | 1.000 | 1.000 | 5874.8 |
| linear_softmax_compression_weighted_dense_control | instruction | 1.000 | 1.000 | 4815.4 |
| linear_softmax_compression_weighted_dense_control | reasoning | 1.000 | 1.000 | 5100.8 |
| linear_softmax_compression_weighted_dense_control | retrieval | 1.000 | 1.000 | 3871.8 |
| linear_softmax_compression_floor_0p95_dense_control | cache | 0.727 | 0.738 | 5087.5 |
| linear_softmax_compression_floor_0p95_dense_control | instruction | 0.618 | 0.627 | 3474.3 |
| linear_softmax_compression_floor_0p95_dense_control | reasoning | 0.638 | 0.641 | 3805.3 |
| linear_softmax_compression_floor_0p95_dense_control | retrieval | 0.698 | 0.740 | 2978.3 |
| linear_softmax_compression_floor_0p90_dense_control | cache | 0.727 | 0.738 | 5087.5 |
| linear_softmax_compression_floor_0p90_dense_control | instruction | 0.618 | 0.627 | 3474.3 |
| linear_softmax_compression_floor_0p90_dense_control | reasoning | 0.638 | 0.641 | 3805.3 |
| linear_softmax_compression_floor_0p90_dense_control | retrieval | 0.698 | 0.740 | 2978.3 |
| linear_softmax_compression_floor_0p85_dense_control | cache | 0.727 | 0.738 | 5087.5 |
| linear_softmax_compression_floor_0p85_dense_control | instruction | 0.618 | 0.627 | 3474.3 |
| linear_softmax_compression_floor_0p85_dense_control | reasoning | 0.701 | 0.715 | 4142.4 |
| linear_softmax_compression_floor_0p85_dense_control | retrieval | 0.738 | 0.847 | 3659.0 |
| linear_softmax_compression_floor_0p80_dense_control | cache | 0.727 | 0.738 | 5087.5 |
| linear_softmax_compression_floor_0p80_dense_control | instruction | 0.790 | 0.872 | 4560.2 |
| linear_softmax_compression_floor_0p80_dense_control | reasoning | 0.778 | 0.852 | 4885.4 |
| linear_softmax_compression_floor_0p80_dense_control | retrieval | 0.625 | 0.871 | 4094.5 |
| linear_softmax_compression_floor_0p75_dense_control | cache | 0.827 | 0.876 | 5706.6 |
| linear_softmax_compression_floor_0p75_dense_control | instruction | 0.728 | 0.770 | 4045.4 |
| linear_softmax_compression_floor_0p75_dense_control | reasoning | 0.827 | 0.866 | 4839.2 |
| linear_softmax_compression_floor_0p75_dense_control | retrieval | 0.724 | 0.919 | 4183.3 |
| linear_softmax_compression_floor_0p70_dense_control | cache | 0.827 | 0.851 | 5530.6 |
| linear_softmax_compression_floor_0p70_dense_control | instruction | 0.655 | 0.668 | 3581.1 |
| linear_softmax_compression_floor_0p70_dense_control | reasoning | 0.777 | 0.797 | 4507.3 |
| linear_softmax_compression_floor_0p70_dense_control | retrieval | 0.776 | 0.883 | 3792.2 |

## By Prompt Variant

| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |
| --- | --- | ---: | ---: | ---: |
| linear_softmax_compression_weighted_dense_control | arithmetic | 1.000 | 1.000 | 5162.0 |
| linear_softmax_compression_weighted_dense_control | bandwidth | 1.000 | 1.000 | 6715.6 |
| linear_softmax_compression_weighted_dense_control | constraints | 1.000 | 1.000 | 4781.8 |
| linear_softmax_compression_weighted_dense_control | formatting | 1.000 | 1.000 | 4901.4 |
| linear_softmax_compression_weighted_dense_control | locality | 1.000 | 1.000 | 4516.3 |
| linear_softmax_compression_weighted_dense_control | logic | 1.000 | 1.000 | 5198.1 |
| linear_softmax_compression_weighted_dense_control | memo | 1.000 | 1.000 | 3850.9 |
| linear_softmax_compression_weighted_dense_control | transcript | 1.000 | 1.000 | 3238.0 |
| linear_softmax_compression_floor_0p95_dense_control | arithmetic | 0.650 | 0.650 | 3932.8 |
| linear_softmax_compression_floor_0p95_dense_control | bandwidth | 0.798 | 0.798 | 5970.3 |
| linear_softmax_compression_floor_0p95_dense_control | constraints | 0.630 | 0.630 | 3430.2 |
| linear_softmax_compression_floor_0p95_dense_control | formatting | 0.578 | 0.609 | 3559.4 |
| linear_softmax_compression_floor_0p95_dense_control | locality | 0.632 | 0.653 | 3640.9 |
| linear_softmax_compression_floor_0p95_dense_control | logic | 0.667 | 0.672 | 3939.9 |
| linear_softmax_compression_floor_0p95_dense_control | memo | 0.760 | 0.760 | 2900.0 |
| linear_softmax_compression_floor_0p95_dense_control | transcript | 0.496 | 0.762 | 3097.2 |
| linear_softmax_compression_floor_0p90_dense_control | arithmetic | 0.650 | 0.650 | 3932.8 |
| linear_softmax_compression_floor_0p90_dense_control | bandwidth | 0.798 | 0.798 | 5970.3 |
| linear_softmax_compression_floor_0p90_dense_control | constraints | 0.630 | 0.630 | 3430.2 |
| linear_softmax_compression_floor_0p90_dense_control | formatting | 0.578 | 0.609 | 3559.4 |
| linear_softmax_compression_floor_0p90_dense_control | locality | 0.632 | 0.653 | 3640.9 |
| linear_softmax_compression_floor_0p90_dense_control | logic | 0.667 | 0.672 | 3939.9 |
| linear_softmax_compression_floor_0p90_dense_control | memo | 0.760 | 0.760 | 2900.0 |
| linear_softmax_compression_floor_0p90_dense_control | transcript | 0.496 | 0.762 | 3097.2 |
| linear_softmax_compression_floor_0p85_dense_control | arithmetic | 0.650 | 0.650 | 3932.8 |
| linear_softmax_compression_floor_0p85_dense_control | bandwidth | 0.798 | 0.798 | 5970.3 |
| linear_softmax_compression_floor_0p85_dense_control | constraints | 0.630 | 0.630 | 3430.2 |
| linear_softmax_compression_floor_0p85_dense_control | formatting | 0.578 | 0.609 | 3559.4 |
| linear_softmax_compression_floor_0p85_dense_control | locality | 0.632 | 0.653 | 3640.9 |
| linear_softmax_compression_floor_0p85_dense_control | logic | 0.752 | 0.771 | 4389.4 |
| linear_softmax_compression_floor_0p85_dense_control | memo | 0.740 | 0.836 | 3582.7 |
| linear_softmax_compression_floor_0p85_dense_control | transcript | 0.650 | 0.969 | 4115.3 |
| linear_softmax_compression_floor_0p80_dense_control | arithmetic | 0.754 | 0.835 | 4918.4 |
| linear_softmax_compression_floor_0p80_dense_control | bandwidth | 0.798 | 0.798 | 5970.3 |
| linear_softmax_compression_floor_0p80_dense_control | constraints | 0.800 | 0.880 | 4542.5 |
| linear_softmax_compression_floor_0p80_dense_control | formatting | 0.766 | 0.891 | 4901.4 |
| linear_softmax_compression_floor_0p80_dense_control | locality | 0.632 | 0.653 | 3640.9 |
| linear_softmax_compression_floor_0p80_dense_control | logic | 0.808 | 0.893 | 5103.3 |
| linear_softmax_compression_floor_0p80_dense_control | memo | 0.665 | 0.884 | 4107.9 |
| linear_softmax_compression_floor_0p80_dense_control | transcript | 0.362 | 0.931 | 4326.5 |
| linear_softmax_compression_floor_0p75_dense_control | arithmetic | 0.801 | 0.849 | 4861.8 |
| linear_softmax_compression_floor_0p75_dense_control | bandwidth | 0.918 | 0.936 | 6549.2 |
| linear_softmax_compression_floor_0p75_dense_control | constraints | 0.700 | 0.750 | 3965.2 |
| linear_softmax_compression_floor_0p75_dense_control | formatting | 0.781 | 0.812 | 4307.4 |
| linear_softmax_compression_floor_0p75_dense_control | locality | 0.702 | 0.802 | 4461.0 |
| linear_softmax_compression_floor_0p75_dense_control | logic | 0.892 | 0.925 | 5138.6 |
| linear_softmax_compression_floor_0p75_dense_control | memo | 0.801 | 0.939 | 4145.5 |
| linear_softmax_compression_floor_0p75_dense_control | transcript | 0.354 | 0.892 | 4055.7 |
| linear_softmax_compression_floor_0p70_dense_control | arithmetic | 0.749 | 0.772 | 4499.7 |
| linear_softmax_compression_floor_0p70_dense_control | bandwidth | 0.918 | 0.936 | 6549.2 |
| linear_softmax_compression_floor_0p70_dense_control | constraints | 0.660 | 0.670 | 3514.6 |
| linear_softmax_compression_floor_0p70_dense_control | formatting | 0.641 | 0.672 | 3779.4 |
| linear_softmax_compression_floor_0p70_dense_control | locality | 0.702 | 0.730 | 3958.2 |
| linear_softmax_compression_floor_0p70_dense_control | logic | 0.864 | 0.878 | 4880.8 |
| linear_softmax_compression_floor_0p70_dense_control | memo | 0.864 | 0.904 | 3765.5 |
| linear_softmax_compression_floor_0p70_dense_control | transcript | 0.346 | 0.862 | 3795.8 |
