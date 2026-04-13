# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 3
- dense avg ms/step: 69.4238
- hand-tuned avg ms/step: 1176.1939
- bias avg ms/step: 1340.0523
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.667
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.1719
- hand-tuned score ms/case: 75.3620
- bias score ms/case: 73.4232
- hand-tuned selection ms/case: 550.9665
- bias selection ms/case: 664.2040
- hand-tuned optional-selection ms/case: 26.5144
- bias optional-selection ms/case: 30.8578
- hand-tuned diverse-selection ms/case: 3.4950
- bias diverse-selection ms/case: 4.9992
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0812
- bias policy-bias ms/case: 0.1380
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

- decode_pseudocode: dense 77.8549 ms/step, hand 1263.6946, bias 1798.4349, hand=dense True, bias=dense True, bias=hand True, hand select 582.64 ms, bias select 881.26 ms
- state_cache_roadmap: dense 66.2395 ms/step, hand 1151.0691, bias 1137.1188, hand=dense True, bias=dense True, bias=hand True, hand select 539.80 ms, bias select 571.50 ms
- statecache_showcase: dense 64.1771 ms/step, hand 1113.8181, bias 1084.6033, hand=dense True, bias=dense True, bias=hand True, hand select 530.46 ms, bias select 539.85 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
