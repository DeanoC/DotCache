# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 1
- dense avg ms/step: 35.7168
- hand-tuned avg ms/step: 430.5184
- bias avg ms/step: 280.5628
- hand-tuned vs dense exact match rate: 0.000
- bias vs dense exact match rate: 0.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.2502
- hand-tuned score ms/case: 187.1408
- bias score ms/case: 151.0381
- hand-tuned selection ms/case: 238.4902
- bias selection ms/case: 167.9708
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
- hand-tuned executed M0 blocks/case: 4032.00
- bias executed M0 blocks/case: 4032.00
- hand-tuned executed M3 blocks/case: 48.00
- bias executed M3 blocks/case: 48.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- state_cache_roadmap: dense 35.7168 ms/step, hand 430.5184, bias 280.5628, hand=dense False, bias=dense False, bias=hand True, hand select 238.49 ms, bias select 167.97 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
