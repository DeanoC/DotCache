# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 1
- dense avg ms/step: 35.9899
- hand-tuned avg ms/step: 852.4878
- bias avg ms/step: 822.3755
- hand-tuned vs dense exact match rate: 0.000
- bias vs dense exact match rate: 0.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.4615
- hand-tuned score ms/case: 126.3442
- bias score ms/case: 115.6425
- hand-tuned selection ms/case: 179.7247
- bias selection ms/case: 140.9864
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

- state_cache_roadmap: dense 35.9899 ms/step, hand 852.4878, bias 822.3755, hand=dense False, bias=dense False, bias=hand True, hand select 179.72 ms, bias select 140.99 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
