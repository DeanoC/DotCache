# Qwen LongBench Selector Compare

| max_prompt_tokens | case | n_rows | mean_exact_match | mean_qa_f1 | mean_decode_ms | p95_decode_ms | mean_ppl_ratio | mean_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | exact | 4 | 0.250 | 0.441 | 627.282 | 637.014 | 0.000 | 0.158 |
| 4096 | quality | 4 | 0.250 | 0.441 | 583.620 | 592.866 | 0.000 | 0.051 |
| 4096 | streaming_sink_recent | 4 | 0.250 | 0.441 | 257.163 | 265.556 | 0.000 | 0.969 |
| 4096 | systems | 4 | 0.250 | 0.441 | 93.728 | 94.490 | 0.000 | 0.023 |
| 8192 | exact | 4 | 0.000 | 0.291 | 1066.429 | 1075.009 | 0.000 | 0.167 |
| 8192 | quality | 4 | 0.000 | 0.291 | 798.470 | 816.988 | 0.000 | 0.047 |
| 8192 | streaming_sink_recent | 4 | 0.000 | 0.291 | 283.477 | 285.764 | 0.000 | 1.016 |
| 8192 | systems | 4 | 0.000 | 0.291 | 159.378 | 163.121 | 0.000 | 0.023 |

## Tradeoff

| max_prompt_tokens | quality_vs_exact_speedup | systems_vs_exact_speedup | systems_vs_quality_speedup | streaming_vs_exact_speedup | quality_minus_systems_exact_match | quality_minus_systems_qa_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1.075 | 6.693 | 6.227 | 2.439 | 0.000 | 0.000 |
| 8192 | 1.336 | 6.691 | 5.010 | 3.762 | 0.000 | 0.000 |

## Sample Outputs

| max_prompt_tokens | prompt | dataset | case | exact_match | qa_f1 | generated |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | hotpot_case_order | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpot_case_order | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpot_case_order | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpot_case_order | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | exact | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | quality | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | streaming_sink_recent | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | systems | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | qasper_ghostvlad | qasper | exact | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | quality | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | streaming_sink_recent | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_ghostvlad | qasper | systems | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | exact | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | quality | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | streaming_sink_recent | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | systems | 1.000 | 1.000 | Ozalj |
| 8192 | hotpot_case_order | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpot_case_order | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpot_case_order | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpot_case_order | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | exact | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | quality | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | streaming_sink_recent | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | systems | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | qasper_ghostvlad | qasper | exact | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | quality | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | streaming_sink_recent | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_ghostvlad | qasper | systems | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | exact | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | quality | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | streaming_sink_recent | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | systems | 0.000 | 0.400 | Ozalj, present day Croatia |
