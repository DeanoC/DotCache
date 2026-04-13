# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 26.2198
- hand-tuned avg ms/step: 333.3872
- bias avg ms/step: 333.3473
- hand-tuned vs dense exact match rate: 0.833
- bias vs dense exact match rate: 0.833
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.333
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.6982
- hand-tuned score ms/case: 115.1481
- bias score ms/case: 113.3249
- hand-tuned selection ms/case: 136.5861
- bias selection ms/case: 130.7638
- hand-tuned optional-selection ms/case: 0.0000
- bias optional-selection ms/case: 0.0000
- hand-tuned diverse-selection ms/case: 0.0000
- bias diverse-selection ms/case: 0.0000
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0000
- bias policy-bias ms/case: 0.0000
- hand-tuned direct-M0 assembly ms/case: 0.0000
- bias direct-M0 assembly ms/case: 0.0000
- hand-tuned direct-M0 query-prep ms/case: 0.0000
- bias direct-M0 query-prep ms/case: 0.0000
- hand-tuned direct-M0 gather ms/case: 0.0000
- bias direct-M0 gather ms/case: 0.0000
- hand-tuned direct-M0 score ms/case: 0.0000
- bias direct-M0 score ms/case: 0.0000
- hand-tuned executed M0 blocks/case: 0.00
- bias executed M0 blocks/case: 0.00
- hand-tuned executed M3 blocks/case: 0.00
- bias executed M3 blocks/case: 0.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- cuda_shortlist_paper_table: dense 35.5315 ms/step, hand 412.9615, bias 405.4926, hand=dense True, bias=dense True, bias=hand True, hand select 170.74 ms, bias select 132.08 ms
- model_roadmap: dense 24.0997 ms/step, hand 314.9806, bias 317.2520, hand=dense True, bias=dense True, bias=hand True, hand select 128.81 ms, bias select 129.33 ms
- real_mixed_probe_code: dense 24.1774 ms/step, hand 314.9064, bias 317.2973, hand=dense True, bias=dense True, bias=hand True, hand select 128.88 ms, bias select 130.24 ms
- serving_policy_compare_code: dense 24.6873 ms/step, hand 400.3478, bias 404.5017, hand=dense True, bias=dense True, bias=hand True, hand select 129.93 ms, bias select 133.47 ms
- stage9_backend_comparison: dense 24.4213 ms/step, hand 232.0799, bias 237.5704, hand=dense True, bias=dense True, bias=hand True, hand select 128.73 ms, bias select 130.43 ms
- submission_execution_plan: dense 24.4013 ms/step, hand 325.0469, bias 317.9698, hand=dense False, bias=dense False, bias=hand True, hand select 132.43 ms, bias select 129.03 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
