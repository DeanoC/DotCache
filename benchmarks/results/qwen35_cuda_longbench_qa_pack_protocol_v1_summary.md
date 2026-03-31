# Qwen3.5 CUDA LongBench QA Pack Summary

| Case | n prompts | Mean QA F1 | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean prompt tokens | Mean selected pages | Datasets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | 4 | 0.14 | 0.00 | 743.41 | 290.39 | 1053.92 | 344.37 | 337.48 | 8520.00 | 0.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_base | 4 | 0.08 | 0.00 | 178.55 | 159.73 | 189.42 | 13.03 | 12.77 | 8520.00 | 65232.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_l23_ctx | 4 | 0.08 | 0.00 | 176.41 | 153.67 | 196.03 | 17.66 | 17.30 | 8520.00 | 65424.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |

## Per-Prompt Rows

| Prompt | Dataset | Row | Case | QA F1 | Exact-match | Prompt tokens | Decode ms/step |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| hotpot_case_order | hotpotqa | 0 | exact | 0.38 | 0.00 | 11818 | 1053.92 |
| hotpot_case_order | hotpotqa | 0 | shortlist_base | 0.12 | 0.00 | 11818 | 189.42 |
| hotpot_case_order | hotpotqa | 0 | shortlist_l23_ctx | 0.12 | 0.00 | 11818 | 196.03 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | exact | 0.20 | 0.00 | 11440 | 962.94 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_base | 0.21 | 0.00 | 11440 | 184.24 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_l23_ctx | 0.21 | 0.00 | 11440 | 174.21 |
| qasper_ghostvlad | qasper | 1 | exact | 0.00 | 0.00 | 3428 | 290.39 |
| qasper_ghostvlad | qasper | 1 | shortlist_base | 0.00 | 0.00 | 3428 | 159.73 |
| qasper_ghostvlad | qasper | 1 | shortlist_l23_ctx | 0.00 | 0.00 | 3428 | 153.67 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | exact | 0.00 | 0.00 | 7394 | 666.39 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_base | 0.00 | 0.00 | 7394 | 180.79 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_l23_ctx | 0.00 | 0.00 | 7394 | 181.73 |
