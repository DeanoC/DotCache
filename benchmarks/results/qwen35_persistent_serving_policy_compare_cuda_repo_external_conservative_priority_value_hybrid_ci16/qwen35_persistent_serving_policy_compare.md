# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 3
- dense avg ms/step: 27.6783
- hand-tuned avg ms/step: 381.9508
- bias avg ms/step: 380.8498
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.333
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 15.2850
- hand-tuned score ms/case: 124.5931
- bias score ms/case: 115.0517
- hand-tuned selection ms/case: 167.2169
- bias selection ms/case: 157.5352
- hand-tuned optional-selection ms/case: 9.2269
- bias optional-selection ms/case: 9.2199
- hand-tuned diverse-selection ms/case: 3.9363
- bias diverse-selection ms/case: 3.9710
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0958
- bias policy-bias ms/case: 0.1293
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

- decode_pseudocode: dense 34.4259 ms/step, hand 389.0554, bias 381.1564, hand=dense True, bias=dense True, bias=hand True, hand select 190.13 ms, bias select 156.77 ms
- state_cache_roadmap: dense 23.8670 ms/step, hand 379.0555, bias 382.0280, hand=dense True, bias=dense True, bias=hand True, hand select 156.06 ms, bias select 159.31 ms
- statecache_showcase: dense 24.7421 ms/step, hand 377.7416, bias 379.3649, hand=dense True, bias=dense True, bias=hand True, hand select 155.46 ms, bias select 156.53 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
