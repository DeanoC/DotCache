# Qwen3.5 CUDA LongBench QA Pack Summary

| Case | n prompts | Mean QA F1 | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean prompt tokens | Mean selected pages | Datasets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | 4 | 0.14 | 0.00 | 743.05 | 286.70 | 1093.96 | 355.11 | 348.01 | 8520.00 | 0.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_base | 4 | 0.08 | 0.00 | 174.91 | 157.17 | 184.07 | 12.43 | 12.19 | 8520.00 | 65232.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_l23_ctx | 4 | 0.08 | 0.00 | 178.86 | 155.23 | 192.10 | 16.77 | 16.43 | 8520.00 | 65424.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_quality_profile | 4 | 0.08 | 0.00 | 186.30 | 168.39 | 198.41 | 12.74 | 12.48 | 8520.00 | 65567.50 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |
| shortlist_topk8 | 4 | 0.08 | 0.00 | 183.53 | 161.86 | 200.23 | 16.00 | 15.68 | 8520.00 | 68304.00 | 2wikimqa, hotpotqa, multifieldqa_en, qasper |

## Per-Prompt Rows

| Prompt | Dataset | Row | Case | QA F1 | Exact-match | Prompt tokens | Decode ms/step |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| hotpot_case_order | hotpotqa | 0 | exact | 0.38 | 0.00 | 11818 | 1093.96 |
| hotpot_case_order | hotpotqa | 0 | shortlist_base | 0.12 | 0.00 | 11818 | 182.97 |
| hotpot_case_order | hotpotqa | 0 | shortlist_l23_ctx | 0.12 | 0.00 | 11818 | 192.10 |
| hotpot_case_order | hotpotqa | 0 | shortlist_quality_profile | 0.12 | 0.00 | 11818 | 198.41 |
| hotpot_case_order | hotpotqa | 0 | shortlist_topk8 | 0.12 | 0.00 | 11818 | 200.23 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | exact | 0.20 | 0.00 | 11440 | 939.55 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_base | 0.21 | 0.00 | 11440 | 175.43 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_l23_ctx | 0.21 | 0.00 | 11440 | 178.76 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_quality_profile | 0.21 | 0.00 | 11440 | 188.06 |
| multifieldqa_isr_transgene | multifieldqa_en | 1 | shortlist_topk8 | 0.20 | 0.00 | 11440 | 184.13 |
| qasper_ghostvlad | qasper | 1 | exact | 0.00 | 0.00 | 3428 | 286.70 |
| qasper_ghostvlad | qasper | 1 | shortlist_base | 0.00 | 0.00 | 3428 | 157.17 |
| qasper_ghostvlad | qasper | 1 | shortlist_l23_ctx | 0.00 | 0.00 | 3428 | 155.23 |
| qasper_ghostvlad | qasper | 1 | shortlist_quality_profile | 0.00 | 0.00 | 3428 | 168.39 |
| qasper_ghostvlad | qasper | 1 | shortlist_topk8 | 0.00 | 0.00 | 3428 | 161.86 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | exact | 0.00 | 0.00 | 7394 | 651.98 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_base | 0.00 | 0.00 | 7394 | 184.07 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_l23_ctx | 0.00 | 0.00 | 7394 | 189.34 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_quality_profile | 0.00 | 0.00 | 7394 | 190.34 |
| wikimqa_spouse_birthplace | 2wikimqa | 0 | shortlist_topk8 | 0.00 | 0.00 | 7394 | 187.89 |
