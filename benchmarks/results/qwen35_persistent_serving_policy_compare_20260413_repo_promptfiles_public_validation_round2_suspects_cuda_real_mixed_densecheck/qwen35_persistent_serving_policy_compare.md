# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 2
- dense avg ms/step: 30.3532
- hand-tuned avg ms/step: 335.5169
- bias avg ms/step: 331.9961
- hand-tuned vs dense exact match rate: 0.000
- bias vs dense exact match rate: 0.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.500
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 20.1588
- hand-tuned score ms/case: 159.7746
- bias score ms/case: 151.5308
- hand-tuned selection ms/case: 197.4094
- bias selection ms/case: 168.6230
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
- hand-tuned executed M0 blocks/case: 5112.00
- bias executed M0 blocks/case: 5112.00
- hand-tuned executed M3 blocks/case: 24.00
- bias executed M3 blocks/case: 24.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- performance_journal: dense 36.7854 ms/step, hand 398.4876, bias 387.4659, hand=dense False, bias=dense False, bias=hand True, hand select 230.81 ms, bias select 172.94 ms
- state_cache_roadmap: dense 23.9210 ms/step, hand 272.5461, bias 276.5262, hand=dense False, bias=dense False, bias=hand True, hand select 164.01 ms, bias select 164.31 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
