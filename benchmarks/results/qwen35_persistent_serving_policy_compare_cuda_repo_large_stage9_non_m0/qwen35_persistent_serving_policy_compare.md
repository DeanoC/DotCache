# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 25.2592
- hand-tuned avg ms/step: 467.8922
- bias avg ms/step: 468.6939
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.200
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 12.3823
- hand-tuned score ms/case: 97.7966
- bias score ms/case: 95.7376
- hand-tuned selection ms/case: 181.5134
- bias selection ms/case: 179.6584
- hand-tuned optional-selection ms/case: 16.0308
- bias optional-selection ms/case: 15.9441
- hand-tuned diverse-selection ms/case: 6.3848
- bias diverse-selection ms/case: 6.3533
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0804
- bias policy-bias ms/case: 0.1103
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
- hand-tuned executed M3 blocks/case: 5289.60
- bias executed M3 blocks/case: 5289.60
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- decode_pseudocode: dense 36.0900 ms/step, hand 198.9390, bias 191.8072, hand=dense True, bias=dense True, bias=hand True, hand select 167.63 ms, bias select 139.82 ms
- layer_revision_tracker: dense 24.1810 ms/step, hand 317.9158, bias 317.8706, hand=dense True, bias=dense True, bias=hand True, hand select 183.20 ms, bias select 182.76 ms
- local_layer_profiles: dense 24.0239 ms/step, hand 315.2968, bias 316.4688, hand=dense False, bias=dense False, bias=hand True, hand select 181.79 ms, bias select 185.55 ms
- model_roadmap: dense 24.1441 ms/step, hand 315.0670, bias 316.8980, hand=dense True, bias=dense True, bias=hand True, hand select 182.05 ms, bias select 182.18 ms
- page_selection_eval: dense 23.9741 ms/step, hand 914.7574, bias 916.8395, hand=dense False, bias=dense False, bias=hand True, hand select 221.70 ms, bias select 222.24 ms
- performance_journal: dense 24.1421 ms/step, hand 1672.0284, bias 1675.5092, hand=dense False, bias=dense False, bias=hand True, hand select 260.20 ms, bias select 263.81 ms
- repo_readme: dense 24.0371 ms/step, hand 314.6851, bias 316.5976, hand=dense False, bias=dense False, bias=hand True, hand select 181.53 ms, bias select 182.05 ms
- state_cache_roadmap: dense 23.9946 ms/step, hand 189.0354, bias 190.8348, hand=dense True, bias=dense True, bias=hand True, hand select 138.34 ms, bias select 138.90 ms
- statecache_showcase: dense 23.9780 ms/step, hand 188.8383, bias 190.6275, hand=dense True, bias=dense True, bias=hand True, hand select 138.10 ms, bias select 138.83 ms
- submission_execution_plan: dense 24.0274 ms/step, hand 252.3584, bias 253.4858, hand=dense False, bias=dense False, bias=hand True, hand select 160.59 ms, bias select 160.45 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
