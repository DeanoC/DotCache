# qwen35_9b LongBench Selector Compare

| max_prompt_tokens | case | n_rows | mean_exact_match | mean_qa_f1 | mean_decode_ms | p95_decode_ms | mean_ppl_ratio | mean_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | exact | 4 | 0.250 | 0.441 | 632.705 | 649.039 | 0.000 | 0.414 |
| 4096 | quality | 4 | 0.250 | 0.441 | 592.300 | 612.570 | 0.000 | 0.385 |
| 4096 | streaming_sink_recent | 4 | 0.250 | 0.441 | 260.797 | 265.246 | 0.000 | 1.046 |
| 4096 | systems | 4 | 0.250 | 0.441 | 96.195 | 97.778 | 0.000 | 0.383 |
| 8192 | exact | 4 | 0.000 | 0.291 | 1046.409 | 1053.647 | 0.000 | 0.361 |
| 8192 | quality | 4 | 0.000 | 0.291 | 787.975 | 800.225 | 0.000 | 0.333 |
| 8192 | streaming_sink_recent | 4 | 0.000 | 0.291 | 274.512 | 278.413 | 0.000 | 1.040 |
| 8192 | systems | 4 | 0.000 | 0.291 | 152.936 | 154.320 | 0.000 | 0.331 |

## Tradeoff

| max_prompt_tokens | quality_vs_exact_speedup | systems_vs_exact_speedup | systems_vs_quality_speedup | streaming_vs_exact_speedup | quality_minus_systems_exact_match | quality_minus_systems_qa_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1.068 | 6.577 | 6.157 | 2.426 | 0.000 | 0.000 |
| 8192 | 1.328 | 6.842 | 5.152 | 3.812 | 0.000 | 0.000 |

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
