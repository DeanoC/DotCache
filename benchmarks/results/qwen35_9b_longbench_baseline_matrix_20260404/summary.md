# Qwen3.5 CUDA LongBench QA Pack Summary

| Case | n prompts | Mean QA F1 | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean prompt tokens | Mean selected pages | Datasets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | 4 | 0.26 | 0.00 | 1612.87 | 673.52 | 2223.20 | 746.21 | 731.29 | 8563.25 | 0.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_base | 4 | 0.10 | 0.00 | 368.36 | 328.45 | 393.69 | 28.41 | 27.84 | 8563.25 | 173952.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_l23_ctx | 4 | 0.10 | 0.00 | 371.37 | 329.93 | 398.49 | 29.29 | 28.70 | 8563.25 | 174336.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_quality_profile | 4 | 0.10 | 0.00 | 369.82 | 324.30 | 393.66 | 32.17 | 31.53 | 8563.25 | 173952.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_topk8 | 4 | 0.32 | 0.00 | 379.23 | 338.93 | 405.00 | 28.70 | 28.12 | 8563.25 | 182144.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |

## Per-Prompt Rows

| Prompt | Dataset | Row | Case | QA F1 | Exact-match | Prompt tokens | Decode ms/step |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| hotpot_case_order | hotpotqa | 0 | exact | 0.15 | 0.00 | 11868 | 2223.20 |
| hotpot_case_order | hotpotqa | 0 | shortlist_base | 0.00 | 0.00 | 11868 | 382.03 |
| hotpot_case_order | hotpotqa | 0 | shortlist_l23_ctx | 0.00 | 0.00 | 11868 | 381.59 |
| hotpot_case_order | hotpotqa | 0 | shortlist_quality_profile | 0.00 | 0.00 | 11868 | 391.37 |
| hotpot_case_order | hotpotqa | 0 | shortlist_topk8 | 0.00 | 0.00 | 11868 | 392.71 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | exact | 0.29 | 0.00 | 11465 | 2202.51 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_base | 0.29 | 0.00 | 11465 | 393.69 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_l23_ctx | 0.29 | 0.00 | 11465 | 398.49 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_quality_profile | 0.29 | 0.00 | 11465 | 393.66 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_topk8 | 0.29 | 0.00 | 11465 | 405.00 |
| qasper_ghostvlad | qasper | 1 | exact | 0.39 | 0.00 | 3476 | 673.52 |
| qasper_ghostvlad | qasper | 1 | shortlist_base | 0.11 | 0.00 | 3476 | 328.45 |
| qasper_ghostvlad | qasper | 1 | shortlist_l23_ctx | 0.11 | 0.00 | 3476 | 329.93 |
| qasper_ghostvlad | qasper | 1 | shortlist_quality_profile | 0.11 | 0.00 | 3476 | 324.30 |
| qasper_ghostvlad | qasper | 1 | shortlist_topk8 | 0.35 | 0.00 | 3476 | 338.93 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | exact | 0.22 | 0.00 | 7444 | 1352.24 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_base | 0.00 | 0.00 | 7444 | 369.27 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_l23_ctx | 0.00 | 0.00 | 7444 | 375.48 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_quality_profile | 0.00 | 0.00 | 7444 | 369.95 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_topk8 | 0.67 | 0.00 | 7444 | 380.27 |
