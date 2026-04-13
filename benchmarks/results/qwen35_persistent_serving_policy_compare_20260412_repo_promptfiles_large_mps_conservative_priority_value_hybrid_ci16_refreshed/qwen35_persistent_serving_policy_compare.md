# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 75.4870
- hand-tuned avg ms/step: 2108.7306
- bias avg ms/step: 2054.4195
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 9.5048
- hand-tuned score ms/case: 111.0873
- bias score ms/case: 66.8099
- hand-tuned selection ms/case: 1134.0578
- bias selection ms/case: 1096.1606
- hand-tuned optional-selection ms/case: 27.3891
- bias optional-selection ms/case: 27.2780
- hand-tuned diverse-selection ms/case: 5.1869
- bias diverse-selection ms/case: 4.6770
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1095
- bias policy-bias ms/case: 0.1472
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

- decode_pseudocode: dense 73.6117 ms/step, hand 1165.2726, bias 1124.4240, hand=dense True, bias=dense True, bias=hand True, hand select 546.83 ms, bias select 556.47 ms
- layer_revision_tracker: dense 77.4383 ms/step, hand 2163.8164, bias 2111.4931, hand=dense True, bias=dense True, bias=hand True, hand select 1158.94 ms, bias select 1097.45 ms
- local_layer_profiles: dense 72.8709 ms/step, hand 2116.9658, bias 2113.5651, hand=dense False, bias=dense False, bias=hand True, hand select 1125.02 ms, bias select 1137.06 ms
- model_roadmap: dense 68.1956 ms/step, hand 2181.6992, bias 2104.6362, hand=dense True, bias=dense True, bias=hand True, hand select 1140.76 ms, bias select 1132.46 ms
- page_selection_eval: dense 91.7767 ms/step, hand 3139.7380, bias 3092.1572, hand=dense False, bias=dense False, bias=hand True, hand select 1772.58 ms, bias select 1658.40 ms
- performance_journal: dense 91.3921 ms/step, hand 4107.8944, bias 4076.4048, hand=dense False, bias=dense False, bias=hand True, hand select 2346.38 ms, bias select 2302.22 ms
- repo_readme: dense 71.0117 ms/step, hand 2126.2030, bias 2054.3672, hand=dense False, bias=dense False, bias=hand True, hand select 1133.28 ms, bias select 1085.41 ms
- state_cache_roadmap: dense 66.0084 ms/step, hand 1142.0178, bias 1103.5446, hand=dense True, bias=dense True, bias=hand True, hand select 547.29 ms, bias select 561.98 ms
- statecache_showcase: dense 58.9462 ms/step, hand 1172.4645, bias 1108.1958, hand=dense True, bias=dense True, bias=hand True, hand select 558.90 ms, bias select 563.86 ms
- submission_execution_plan: dense 83.6185 ms/step, hand 1771.2346, bias 1655.4070, hand=dense False, bias=dense False, bias=hand True, hand select 1010.60 ms, bias select 866.28 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
