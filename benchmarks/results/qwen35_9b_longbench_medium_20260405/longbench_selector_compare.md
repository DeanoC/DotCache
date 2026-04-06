# Qwen/Qwen3.5-9B LongBench medium Pack Compare

| max_prompt_tokens | case | n_rows | mean_exact_match | mean_qa_f1 | mean_decode_ms | p95_decode_ms | mean_ppl_ratio | mean_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | exact | 12 | 0.167 | 0.270 | 614.239 | 625.644 | 1.012 | 0.460 |
| 4096 | quality | 12 | 0.167 | 0.270 | 574.427 | 588.542 | 1.013 | 0.433 |
| 4096 | streaming_sink_recent | 12 | 0.167 | 0.270 | 257.917 | 262.833 | 1.307 | 0.810 |
| 4096 | systems | 12 | 0.167 | 0.270 | 91.621 | 94.274 | 1.012 | 0.431 |
| 8192 | exact | 12 | 0.167 | 0.280 | 981.825 | 990.493 | 1.023 | 0.429 |
| 8192 | quality | 12 | 0.167 | 0.280 | 773.764 | 780.519 | 1.020 | 0.402 |
| 8192 | streaming_sink_recent | 12 | 0.167 | 0.280 | 278.335 | 280.899 | 1.345 | 0.792 |
| 8192 | systems | 12 | 0.167 | 0.280 | 145.521 | 148.122 | 1.021 | 0.401 |

## Tradeoff

| max_prompt_tokens | quality_vs_exact_speedup | systems_vs_exact_speedup | systems_vs_quality_speedup | streaming_vs_exact_speedup | quality_minus_systems_exact_match | quality_minus_systems_qa_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1.069 | 6.704 | 6.270 | 2.382 | 0.000 | 0.000 |
| 8192 | 1.269 | 6.747 | 5.317 | 3.527 | 0.000 | 0.000 |

## Sample Outputs

| max_prompt_tokens | prompt | dataset | case | exact_match | qa_f1 | generated |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 2wikimqa_row0 | 2wikimqa | exact | 1.000 | 1.000 | Ozalj |
| 4096 | 2wikimqa_row0 | 2wikimqa | quality | 1.000 | 1.000 | Ozalj |
| 4096 | 2wikimqa_row0 | 2wikimqa | streaming_sink_recent | 1.000 | 1.000 | Ozalj |
| 4096 | 2wikimqa_row0 | 2wikimqa | systems | 1.000 | 1.000 | Ozalj |
| 4096 | 2wikimqa_row1 | 2wikimqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row1 | 2wikimqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row1 | 2wikimqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row1 | 2wikimqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row2 | 2wikimqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row2 | 2wikimqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row2 | 2wikimqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | 2wikimqa_row2 | 2wikimqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 4096 | hotpotqa_row0 | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row0 | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row0 | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row0 | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row1 | hotpotqa | exact | 0.000 | 0.286 | Charles L. Clifford The actor that plays Phileas Fogg in "Around the World in 80 Days" is Jay Sheffield (Passage |
| 4096 | hotpotqa_row1 | hotpotqa | quality | 0.000 | 0.286 | Charles L. Clifford The actor that plays Phileas Fogg in "Around the World in 80 Days" is Jay Sheffield (Passage |
| 4096 | hotpotqa_row1 | hotpotqa | streaming_sink_recent | 0.000 | 0.286 | Charles L. Clifford The actor that plays Phileas Fogg in "Around the World in 80 Days" is Jay Sheffield (Passage |
| 4096 | hotpotqa_row1 | hotpotqa | systems | 0.000 | 0.286 | Charles L. Clifford The actor that plays Phileas Fogg in "Around the World in 80 Days" is Jay Sheffield (Passage |
| 4096 | hotpotqa_row2 | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row2 | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row2 | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | hotpotqa_row2 | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, Passage |
| 4096 | multifieldqa_en_row0 | multifieldqa_en | exact | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 4096 | multifieldqa_en_row0 | multifieldqa_en | quality | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 4096 | multifieldqa_en_row0 | multifieldqa_en | streaming_sink_recent | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 4096 | multifieldqa_en_row0 | multifieldqa_en | systems | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 4096 | multifieldqa_en_row1 | multifieldqa_en | exact | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_en_row1 | multifieldqa_en | quality | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_en_row1 | multifieldqa_en | streaming_sink_recent | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_en_row1 | multifieldqa_en | systems | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 4096 | multifieldqa_en_row2 | multifieldqa_en | exact | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 4096 | multifieldqa_en_row2 | multifieldqa_en | quality | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 4096 | multifieldqa_en_row2 | multifieldqa_en | streaming_sink_recent | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 4096 | multifieldqa_en_row2 | multifieldqa_en | systems | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 4096 | qasper_row0 | qasper | exact | 0.000 | 0.000 | unanswerable |
| 4096 | qasper_row0 | qasper | quality | 0.000 | 0.000 | unanswerable |
| 4096 | qasper_row0 | qasper | streaming_sink_recent | 0.000 | 0.000 | unanswerable |
| 4096 | qasper_row0 | qasper | systems | 0.000 | 0.000 | unanswerable |
| 4096 | qasper_row1 | qasper | exact | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_row1 | qasper | quality | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_row1 | qasper | streaming_sink_recent | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_row1 | qasper | systems | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 4096 | qasper_row2 | qasper | exact | 0.000 | 0.125 | 68.8% to 71.8% |
| 4096 | qasper_row2 | qasper | quality | 0.000 | 0.125 | 68.8% to 71.8% |
| 4096 | qasper_row2 | qasper | streaming_sink_recent | 0.000 | 0.125 | 68.8% to 71.8% |
| 4096 | qasper_row2 | qasper | systems | 0.000 | 0.125 | 68.8% to 71.8% |
| 8192 | 2wikimqa_row0 | 2wikimqa | exact | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | 2wikimqa_row0 | 2wikimqa | quality | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | 2wikimqa_row0 | 2wikimqa | streaming_sink_recent | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | 2wikimqa_row0 | 2wikimqa | systems | 0.000 | 0.400 | Ozalj, present day Croatia |
| 8192 | 2wikimqa_row1 | 2wikimqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: 10 passages about various people (Jim Ram |
| 8192 | 2wikimqa_row1 | 2wikimqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: 10 passages about various people (Jim Ram |
| 8192 | 2wikimqa_row1 | 2wikimqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: 10 passages about various people (Jim Ram |
| 8192 | 2wikimqa_row1 | 2wikimqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: 10 passages about various people (Jim Ram |
| 8192 | 2wikimqa_row2 | 2wikimqa | exact | 0.000 | 0.000 | Paris |
| 8192 | 2wikimqa_row2 | 2wikimqa | quality | 0.000 | 0.000 | Paris |
| 8192 | 2wikimqa_row2 | 2wikimqa | streaming_sink_recent | 0.000 | 0.000 | Paris |
| 8192 | 2wikimqa_row2 | 2wikimqa | systems | 0.000 | 0.000 | Paris |
| 8192 | hotpotqa_row0 | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpotqa_row0 | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpotqa_row0 | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpotqa_row0 | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1 to Passage |
| 8192 | hotpotqa_row1 | hotpotqa | exact | 1.000 | 1.000 | Charles L. Clifford |
| 8192 | hotpotqa_row1 | hotpotqa | quality | 1.000 | 1.000 | Charles L. Clifford |
| 8192 | hotpotqa_row1 | hotpotqa | streaming_sink_recent | 1.000 | 1.000 | Charles L. Clifford |
| 8192 | hotpotqa_row1 | hotpotqa | systems | 1.000 | 1.000 | Charles L. Clifford |
| 8192 | hotpotqa_row2 | hotpotqa | exact | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, 2 |
| 8192 | hotpotqa_row2 | hotpotqa | quality | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, 2 |
| 8192 | hotpotqa_row2 | hotpotqa | streaming_sink_recent | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, 2 |
| 8192 | hotpotqa_row2 | hotpotqa | systems | 0.000 | 0.000 | <think> 1. **Analyze the Request:** * Input: Several passages (Passage 1, 2 |
| 8192 | multifieldqa_en_row0 | multifieldqa_en | exact | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 8192 | multifieldqa_en_row0 | multifieldqa_en | quality | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 8192 | multifieldqa_en_row0 | multifieldqa_en | streaming_sink_recent | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 8192 | multifieldqa_en_row0 | multifieldqa_en | systems | 0.000 | 0.067 | <think> 1. **Analyze the Request:** * Input: A text about Football Club Urartu (formerly FC Banants). * Task: Answer a specific question based on the text. * Question: "What is the name of the |
| 8192 | multifieldqa_en_row1 | multifieldqa_en | exact | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_en_row1 | multifieldqa_en | quality | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_en_row1 | multifieldqa_en | streaming_sink_recent | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_en_row1 | multifieldqa_en | systems | 0.000 | 0.375 | No, the ISR is neither sufficient nor necessary to induce transgene reactivation |
| 8192 | multifieldqa_en_row2 | multifieldqa_en | exact | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 8192 | multifieldqa_en_row2 | multifieldqa_en | quality | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 8192 | multifieldqa_en_row2 | multifieldqa_en | streaming_sink_recent | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 8192 | multifieldqa_en_row2 | multifieldqa_en | systems | 1.000 | 1.000 | Low temperature scanning tunneling microscopy and spectroscopy (STM/STS) |
| 8192 | qasper_row0 | qasper | exact | 0.000 | 0.000 | unanswerable |
| 8192 | qasper_row0 | qasper | quality | 0.000 | 0.000 | unanswerable |
| 8192 | qasper_row0 | qasper | streaming_sink_recent | 0.000 | 0.000 | unanswerable |
| 8192 | qasper_row0 | qasper | systems | 0.000 | 0.000 | unanswerable |
| 8192 | qasper_row1 | qasper | exact | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_row1 | qasper | quality | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_row1 | qasper | streaming_sink_recent | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_row1 | qasper | systems | 0.000 | 0.389 | GhostVLAD is an extension of the NetVLAD approach that adds ghost clusters to map noisy or irrelevant content into ghost clusters, which are excluded during the feature aggregation stage |
| 8192 | qasper_row2 | qasper | exact | 0.000 | 0.125 | 68.8% to 71.8% |
| 8192 | qasper_row2 | qasper | quality | 0.000 | 0.125 | 68.8% to 71.8% |
| 8192 | qasper_row2 | qasper | streaming_sink_recent | 0.000 | 0.125 | 68.8% to 71.8% |
| 8192 | qasper_row2 | qasper | systems | 0.000 | 0.125 | 68.8% to 71.8% |
