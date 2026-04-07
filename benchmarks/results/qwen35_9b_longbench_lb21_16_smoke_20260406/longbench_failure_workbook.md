# Qwen/Qwen3.5-9B LongBench Failure Workbook

## Summary

| classification | n_rows |
| --- | --- |
| downstream_under_attention | 6 |

## Task Family Breakdown

| task_family | classification | n_rows |
| --- | --- | --- |
| code | downstream_under_attention | 2 |
| qa | downstream_under_attention | 2 |
| summarization | downstream_under_attention | 2 |

## Workbook

| max_prompt_tokens | prompt | dataset | task_family | classification | exact_score | systems_score | systems_raw_score | gap_vs_exact | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | dureader_row117 | dureader | qa | downstream_under_attention | 0.208 | 0.206 | 0.194 | 0.001 | systems still trails exact by 0.001 after cleaning, but the miss is not explained by formatting alone |
| 4096 | repobench-p_row464 | repobench-p | code | downstream_under_attention | 0.361 | 0.359 | 0.690 | 0.001 | systems still trails exact by 0.001 after cleaning, but the miss is not explained by formatting alone |
| 8192 | dureader_row117 | dureader | qa | downstream_under_attention | 0.271 | 0.269 | 0.253 | 0.002 | systems still trails exact by 0.002 after cleaning, but the miss is not explained by formatting alone |
| 8192 | multi_news_row72 | multi_news | summarization | downstream_under_attention | 0.166 | 0.166 | 0.166 | 0.000 | systems still trails exact by 0.000 after cleaning, but the miss is not explained by formatting alone |
| 8192 | qmsum_row88 | qmsum | summarization | downstream_under_attention | 0.092 | 0.091 | 0.087 | 0.001 | systems still trails exact by 0.001 after cleaning, but the miss is not explained by formatting alone |
| 8192 | repobench-p_row464 | repobench-p | code | downstream_under_attention | 0.361 | 0.359 | 0.690 | 0.001 | systems still trails exact by 0.001 after cleaning, but the miss is not explained by formatting alone |
