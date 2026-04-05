# qwen35_4b LongBench Selector Compare

| max_prompt_tokens | case | n_rows | mean_exact_match | mean_qa_f1 | mean_decode_ms | p95_decode_ms | mean_ppl_ratio | mean_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | exact | 4 | 0.000 | 0.253 | 625.448 | 633.926 | 0.000 | 0.519 |
| 4096 | quality | 4 | 0.000 | 0.253 | 1165.355 | 1210.864 | 0.000 | 0.496 |
| 4096 | streaming_sink_recent | 4 | 0.000 | 0.253 | 258.878 | 270.252 | 0.000 | 1.203 |
| 4096 | systems | 4 | 0.000 | 0.253 | 373.600 | 388.383 | 0.000 | 0.491 |
| 8192 | exact | 4 | 0.000 | 0.247 | 1051.394 | 1069.017 | 0.000 | 0.519 |
| 8192 | quality | 4 | 0.000 | 0.247 | 2147.806 | 2186.643 | 0.000 | 0.532 |
| 8192 | streaming_sink_recent | 4 | 0.000 | 0.247 | 271.088 | 275.845 | 0.000 | 1.189 |
| 8192 | systems | 4 | 0.000 | 0.247 | 800.082 | 806.332 | 0.000 | 0.534 |

## Tradeoff

| max_prompt_tokens | quality_vs_exact_speedup | systems_vs_exact_speedup | systems_vs_quality_speedup | streaming_vs_exact_speedup | quality_minus_systems_exact_match | quality_minus_systems_qa_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 0.537 | 1.674 | 3.119 | 2.416 | 0.000 | 0.000 |
| 8192 | 0.490 | 1.314 | 2.684 | 3.878 | 0.000 | 0.000 |

## Sample Outputs

| max_prompt_tokens | prompt | dataset | case | exact_match | qa_f1 | generated |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | hotpot_case_order | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 4096 | hotpot_case_order | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 4096 | hotpot_case_order | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 4096 | hotpot_case_order | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | exact | 0.000 | 0.227 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon Salubrinal treatment, it does not induce OA1 transgene reactivation. Furthermore, treatment with ISRIB, which inhibits the ISR, does not affect the reactivation of the OA1 transgene |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | quality | 0.000 | 0.227 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon Salubrinal treatment, it does not induce OA1 transgene reactivation. Furthermore, treatment with ISRIB, which inhibits the ISR, does not affect the reactivation of the OA1 transgene |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | streaming_sink_recent | 0.000 | 0.227 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon Salubrinal treatment, it does not induce OA1 transgene reactivation. Furthermore, treatment with ISRIB, which inhibits the ISR, does not affect the reactivation of the OA1 transgene |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | systems | 0.000 | 0.227 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon Salubrinal treatment, it does not induce OA1 transgene reactivation. Furthermore, treatment with ISRIB, which inhibits the ISR, does not affect the reactivation of the OA1 transgene |
| 4096 | qasper_ghostvlad | qasper | exact | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | quality | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | streaming_sink_recent | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | systems | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | exact | 0.000 | 0.400 | Ozalj, present day Croatia |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | quality | 0.000 | 0.400 | Ozalj, present day Croatia |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | streaming_sink_recent | 0.000 | 0.400 | Ozalj, present day Croatia |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | systems | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | hotpot_case_order | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 8192 | hotpot_case_order | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 8192 | hotpot_case_order | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 8192 | hotpot_case_order | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Task: Answer the question based on the given passages |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | exact | 0.000 | 0.204 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon EAA starvation, the upregulation of its downstream effector CHOP only partly correlates with transgene reactivation and may not be sufficient to induce it. Furthermore, experiments using ISRIB (which inhib |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | quality | 0.000 | 0.204 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon EAA starvation, the upregulation of its downstream effector CHOP only partly correlates with transgene reactivation and may not be sufficient to induce it. Furthermore, experiments using ISRIB (which inhib |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | streaming_sink_recent | 0.000 | 0.204 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon EAA starvation, the upregulation of its downstream effector CHOP only partly correlates with transgene reactivation and may not be sufficient to induce it. Furthermore, experiments using ISRIB (which inhib |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | systems | 0.000 | 0.204 | No, the ISR is not necessary for transgene reactivation. The text states that while the ISR is activated upon EAA starvation, the upregulation of its downstream effector CHOP only partly correlates with transgene reactivation and may not be sufficient to induce it. Furthermore, experiments using ISRIB (which inhib |
| 8192 | qasper_ghostvlad | qasper | exact | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | quality | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | streaming_sink_recent | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | systems | 0.000 | 0.383 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to absorb noisy or irrelevant content, which are then excluded during the feature aggregation stage |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | exact | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | quality | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | streaming_sink_recent | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | systems | 0.000 | 0.400 | Ozalj, present day Croatia |
