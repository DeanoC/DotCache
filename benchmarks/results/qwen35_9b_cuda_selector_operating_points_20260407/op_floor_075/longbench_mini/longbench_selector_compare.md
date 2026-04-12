# Qwen/Qwen3.5-9B LongBench longbench_mini op_floor_075

| max_prompt_tokens | case | n_rows | mean_official_score | mean_matches_dense_output | mean_exact_match | mean_qa_f1 | mean_decode_ms | p95_decode_ms | mean_eff_bytes_per_tok | mean_ppl_ratio | mean_rmse | worst_dataset_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | dense | 4 | - | 1.000 | 0.250 | 0.524 | 33.411 | 33.411 | 18629.431 | 1.000 | 0.000 | - |
| 4096 | exact | 4 | 0.524 | 1.000 | 0.250 | 0.524 | 896.864 | 896.864 | 18629.431 | 1.011 | 0.326 | 0.333 |
| 4096 | quality | 4 | 0.556 | 0.500 | 0.250 | 0.556 | 3855.343 | 3855.343 | 31676.202 | 1.005 | 0.318 | 0.333 |
| 4096 | systems | 4 | 0.524 | 0.750 | 0.250 | 0.524 | 919.599 | 919.599 | 34225.542 | 1.001 | 0.315 | 0.333 |
| 8192 | dense | 4 | - | 1.000 | 0.000 | 0.374 | 34.573 | 34.573 | 18556.945 | 1.000 | 0.000 | - |
| 8192 | exact | 4 | 0.406 | 0.750 | 0.000 | 0.406 | 1480.983 | 1480.983 | 18556.945 | 1.015 | 0.357 | 0.333 |
| 8192 | quality | 4 | 0.406 | 0.500 | 0.000 | 0.406 | 5555.774 | 5555.774 | 28705.153 | 1.011 | 0.347 | 0.333 |
| 8192 | systems | 4 | 0.406 | 0.500 | 0.000 | 0.406 | 3034.170 | 3034.170 | 31139.535 | 1.013 | 0.344 | 0.333 |

## Tradeoff

| max_prompt_tokens | exact_vs_dense_speedup | quality_vs_dense_speedup | systems_vs_dense_speedup | streaming_vs_dense_speedup | quest_vs_dense_speedup | quality_vs_exact_speedup | systems_vs_exact_speedup | streaming_vs_exact_speedup | quest_vs_exact_speedup | systems_vs_quality_speedup | exact_matches_dense_output | quality_matches_dense_output | systems_matches_dense_output | streaming_matches_dense_output | quest_matches_dense_output | quality_minus_systems_official_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 0.037 | 0.009 | 0.036 | - | - | 0.233 | 0.975 | - | - | 4.192 | 1.000 | 0.500 | 0.750 | - | - | 0.031 |
| 8192 | 0.023 | 0.006 | 0.011 | - | - | 0.267 | 0.488 | - | - | 1.831 | 0.750 | 0.500 | 0.500 | - | - | 0.000 |

## Task Family Breakdown

| max_prompt_tokens | case | task_family | n_rows | mean_official_score | mean_matches_dense_output | mean_decode_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | dense | qa | 4 | - | 1.000 | 33.411 |
| 4096 | exact | qa | 4 | 0.524 | 1.000 | 896.864 |
| 4096 | quality | qa | 4 | 0.556 | 0.500 | 3855.343 |
| 4096 | systems | qa | 4 | 0.524 | 0.750 | 919.599 |
| 8192 | dense | qa | 4 | - | 1.000 | 34.573 |
| 8192 | exact | qa | 4 | 0.406 | 0.750 | 1480.983 |
| 8192 | quality | qa | 4 | 0.406 | 0.500 | 5555.774 |
| 8192 | systems | qa | 4 | 0.406 | 0.500 | 3034.170 |

## Parity

| max_prompt_tokens | match_mode | external_case | systems_official_score | external_official_score | official_gap | systems_eff_bytes | external_eff_bytes | eff_bytes_gap | systems_vs_external_speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Confidence

| max_prompt_tokens | comparison_case | mean_delta | ci_low | ci_high | n_datasets | win_datasets | loss_datasets | tie_datasets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | exact | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |
| 4096 | quality | -0.031 | -0.094 | 0.000 | 4 | 0 | 1 | 3 |
| 8192 | exact | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |
| 8192 | quality | 0.000 | 0.000 | 0.000 | 4 | 0 | 0 | 4 |

## Sample Outputs

| max_prompt_tokens | prompt | dataset | task_family | case | official_score | matches_dense_output | exact_match | qa_f1 | generated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | hotpot_case_order | hotpotqa | qa | dense | - | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 4096 | hotpot_case_order | hotpotqa | qa | exact | 0.333 | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 4096 | hotpot_case_order | hotpotqa | qa | quality | 0.333 | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 4096 | hotpot_case_order | hotpotqa | qa | systems | 0.333 | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | qa | dense | - | 1.000 | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | qa | exact | 0.375 | 1.000 | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | qa | quality | 0.500 | 0.000 | 0.000 | 0.500 | No, the ISR is not sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | qa | systems | 0.375 | 1.000 | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | qasper_ghostvlad | qasper | qa | dense | - | 1.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | qa | exact | 0.389 | 1.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | qa | quality | 0.389 | 0.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds Ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | qa | systems | 0.389 | 0.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds Ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | qa | dense | - | 1.000 | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | qa | exact | 1.000 | 1.000 | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | qa | quality | 1.000 | 1.000 | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | qa | systems | 1.000 | 1.000 | 1.000 | 1.000 | Ozalj |
| 8192 | hotpot_case_order | hotpotqa | qa | dense | - | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 8192 | hotpot_case_order | hotpotqa | qa | exact | 0.333 | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 8192 | hotpot_case_order | hotpotqa | qa | quality | 0.333 | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 8192 | hotpot_case_order | hotpotqa | qa | systems | 0.333 | 1.000 | 0.000 | 0.333 | Gates v. Collier |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | qa | dense | - | 1.000 | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | qa | exact | 0.500 | 0.000 | 0.000 | 0.500 | No, the ISR is not sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | qa | quality | 0.500 | 0.000 | 0.000 | 0.500 | No, the ISR is not sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | qa | systems | 0.500 | 0.000 | 0.000 | 0.500 | No, the ISR is not sufficient nor necessary to induce transgene reactivation |
| 8192 | qasper_ghostvlad | qasper | qa | dense | - | 1.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | qa | exact | 0.389 | 1.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | qa | quality | 0.389 | 0.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds Ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | qa | systems | 0.389 | 0.000 | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds Ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | qa | dense | - | 1.000 | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | qa | exact | 0.400 | 1.000 | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | qa | quality | 0.400 | 1.000 | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | qa | systems | 0.400 | 1.000 | 0.000 | 0.400 | Ozalj, present day Croatia |
