# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 68.4329
- hand-tuned avg ms/step: 1471.1836
- bias avg ms/step: 1434.7838
- hand-tuned vs dense exact match rate: 0.833
- bias vs dense exact match rate: 0.833
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.833
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 8.1676
- hand-tuned score ms/case: 85.4900
- bias score ms/case: 58.9636
- hand-tuned selection ms/case: 790.2556
- bias selection ms/case: 785.7950
- hand-tuned optional-selection ms/case: 24.1130
- bias optional-selection ms/case: 23.7128
- hand-tuned diverse-selection ms/case: 4.3136
- bias diverse-selection ms/case: 4.1760
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0689
- bias policy-bias ms/case: 0.0916
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

- cuda_shortlist_paper_table: dense 71.9696 ms/step, hand 1828.4462, bias 1756.5727, hand=dense True, bias=dense True, bias=hand True, hand select 1001.90 ms, bias select 981.68 ms
- model_roadmap: dense 66.7558 ms/step, hand 1389.1437, bias 1356.5467, hand=dense True, bias=dense True, bias=hand True, hand select 777.20 ms, bias select 736.50 ms
- real_mixed_probe_code: dense 61.4964 ms/step, hand 1349.6840, bias 1350.7291, hand=dense True, bias=dense True, bias=hand True, hand select 709.41 ms, bias select 750.62 ms
- serving_policy_compare_code: dense 67.2566 ms/step, hand 1800.6294, bias 1743.4349, hand=dense True, bias=dense True, bias=hand True, hand select 962.83 ms, bias select 981.27 ms
- stage9_backend_comparison: dense 66.8478 ms/step, hand 968.1068, bias 940.0700, hand=dense True, bias=dense True, bias=hand True, hand select 530.69 ms, bias select 494.16 ms
- submission_execution_plan: dense 76.2712 ms/step, hand 1491.0912, bias 1461.3493, hand=dense False, bias=dense False, bias=hand True, hand select 759.50 ms, bias select 770.54 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
