# qwen35_27b LongBench Selector Compare

| max_prompt_tokens | case | n_rows | mean_exact_match | mean_qa_f1 | mean_decode_ms | p95_decode_ms | mean_ppl_ratio | mean_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | exact | 4 | 0.250 | 0.358 | 1263.516 | 1273.398 | 0.000 | 0.521 |
| 4096 | quality | 4 | 0.250 | 0.358 | 1737.076 | 1761.137 | 0.000 | 0.513 |
| 4096 | streaming_sink_recent | 4 | 0.250 | 0.358 | 538.435 | 542.084 | 0.000 | 0.779 |
| 4096 | systems | 4 | 0.250 | 0.358 | 331.793 | 335.305 | 0.000 | 0.511 |
| 8192 | exact | 4 | 0.250 | 0.341 | 2096.277 | 2117.435 | 0.000 | 0.564 |
| 8192 | quality | 4 | 0.250 | 0.341 | 3512.227 | 3581.215 | 0.000 | 0.550 |
| 8192 | streaming_sink_recent | 4 | 0.250 | 0.341 | 566.767 | 569.926 | 0.000 | 0.948 |
| 8192 | systems | 4 | 0.250 | 0.341 | 804.331 | 813.295 | 0.000 | 0.550 |

## Tradeoff

| max_prompt_tokens | quality_vs_exact_speedup | systems_vs_exact_speedup | systems_vs_quality_speedup | streaming_vs_exact_speedup | quality_minus_systems_exact_match | quality_minus_systems_qa_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 0.727 | 3.808 | 5.235 | 2.347 | 0.000 | 0.000 |
| 8192 | 0.597 | 2.606 | 4.367 | 3.699 | 0.000 | 0.000 |

## Sample Outputs

| max_prompt_tokens | prompt | dataset | case | exact_match | qa_f1 | generated |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | hotpot_case_order | hotpotqa | exact | 0.000 | 0.333 | Gates v. Collier |
| 4096 | hotpot_case_order | hotpotqa | quality | 0.000 | 0.333 | Gates v. Collier |
| 4096 | hotpot_case_order | hotpotqa | streaming_sink_recent | 0.000 | 0.333 | Gates v. Collier |
| 4096 | hotpot_case_order | hotpotqa | systems | 0.000 | 0.333 | Gates v. Collier |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | exact | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A scientific text (likely an abstract and parts of a results/discussion section from a paper). * Task: Answer a specific question based on the text. * Question: "Is |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | quality | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A scientific text (likely an abstract and parts of a results/discussion section from a paper). * Task: Answer a specific question based on the text. * Question: "Is |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | streaming_sink_recent | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A scientific text (likely an abstract and parts of a results/discussion section from a paper). * Task: Answer a specific question based on the text. * Question: "Is |
| 4096 | multifieldqa_isr_transgene | multifieldqa_en | systems | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A scientific text (likely an abstract and parts of a results/discussion section from a paper). * Task: Answer a specific question based on the text. * Question: "Is |
| 4096 | qasper_ghostvlad | qasper | exact | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 4096 | qasper_ghostvlad | qasper | quality | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 4096 | qasper_ghostvlad | qasper | streaming_sink_recent | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 4096 | qasper_ghostvlad | qasper | systems | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | exact | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | quality | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | streaming_sink_recent | 1.000 | 1.000 | Ozalj |
| 4096 | wikimqa_spouse_birthplace | 2wikimqa | systems | 1.000 | 1.000 | Ozalj |
| 8192 | hotpot_case_order | hotpotqa | exact | 0.000 | 0.333 | Gates v. Collier |
| 8192 | hotpot_case_order | hotpotqa | quality | 0.000 | 0.333 | Gates v. Collier |
| 8192 | hotpot_case_order | hotpotqa | streaming_sink_recent | 0.000 | 0.333 | Gates v. Collier |
| 8192 | hotpot_case_order | hotpotqa | systems | 0.000 | 0.333 | Gates v. Collier |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: A scientific text (research article excerpt) about amino acid deprivation, transgene reactivation, and signaling pathways (GCN2, ISR, mTOR, etc.). * Task: Answer a specific |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: A scientific text (research article excerpt) about amino acid deprivation, transgene reactivation, and signaling pathways (GCN2, ISR, mTOR, etc.). * Task: Answer a specific |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: A scientific text (research article excerpt) about amino acid deprivation, transgene reactivation, and signaling pathways (GCN2, ISR, mTOR, etc.). * Task: Answer a specific |
| 8192 | multifieldqa_isr_transgene | multifieldqa_en | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: A scientific text (research article excerpt) about amino acid deprivation, transgene reactivation, and signaling pathways (GCN2, ISR, mTOR, etc.). * Task: Answer a specific |
| 8192 | qasper_ghostvlad | qasper | exact | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 8192 | qasper_ghostvlad | qasper | quality | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 8192 | qasper_ghostvlad | qasper | streaming_sink_recent | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 8192 | qasper_ghostvlad | qasper | systems | 0.000 | 0.032 | <think> 1. **Analyze the Request:** * Input: A scientific article about language identification using Ghost-VLAD pooling. * Task: Answer a specific question based *only* on the article. * Constraints: Concise (single phrase or sentence), no explanation, no role labels, no reasoning, start immediately with the answer. If unanswerable, write "unanswerable". If yes/no, answer "yes"/"no"/"unanswerable". * Question: "What is the GhostVLAD approach?" 2 |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | exact | 1.000 | 1.000 | Ozalj |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | quality | 1.000 | 1.000 | Ozalj |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | streaming_sink_recent | 1.000 | 1.000 | Ozalj |
| 8192 | wikimqa_spouse_birthplace | 2wikimqa | systems | 1.000 | 1.000 | Ozalj |
