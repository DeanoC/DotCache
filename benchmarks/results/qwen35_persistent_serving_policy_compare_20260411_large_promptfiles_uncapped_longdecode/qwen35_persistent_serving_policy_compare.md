# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 80.4025
- hand-tuned avg ms/step: 2732.8306
- bias avg ms/step: 2612.5771
- hand-tuned vs dense exact match rate: 0.300
- bias vs dense exact match rate: 0.300
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 8.7296
- hand-tuned score ms/case: 137.5000
- bias score ms/case: 52.5960
- hand-tuned selection ms/case: 1015.5474
- bias selection ms/case: 924.2872
- hand-tuned optional-selection ms/case: 26.7285
- bias optional-selection ms/case: 25.6757
- hand-tuned diverse-selection ms/case: 4.1506
- bias diverse-selection ms/case: 4.4158
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0842
- bias policy-bias ms/case: 0.1141
- hand-tuned direct-M0 assembly ms/case: 0.0000
- bias direct-M0 assembly ms/case: 0.0000
- hand-tuned direct-M0 score ms/case: 0.0000
- bias direct-M0 score ms/case: 0.0000
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- AAE_DotCache_Robust_Rewrite: dense 92.8480 ms/step, hand 5027.2587, bias 4946.3246, hand=dense True, bias=dense True, bias=hand True, hand select 1780.83 ms, bias select 1756.58 ms
- Minimal Agent Loop: dense 80.8456 ms/step, hand 4175.6396, bias 4115.5129, hand=dense True, bias=dense True, bias=hand True, hand select 1555.04 ms, bias select 1508.23 ms
- auth-and-boundaries: dense 69.5943 ms/step, hand 840.7244, bias 737.1519, hand=dense False, bias=dense False, bias=hand True, hand select 313.62 ms, bias select 222.77 ms
- codex_algorithmic_alternatives: dense 87.2725 ms/step, hand 4355.2762, bias 4272.0378, hand=dense False, bias=dense False, bias=hand True, hand select 1572.93 ms, bias select 1551.52 ms
- external-worker-flow: dense 73.7526 ms/step, hand 1010.3945, bias 880.4463, hand=dense False, bias=dense False, bias=hand True, hand select 370.95 ms, bias select 283.59 ms
- model_roadmap: dense 96.6129 ms/step, hand 6561.0458, bias 6399.3938, hand=dense False, bias=dense False, bias=hand True, hand select 2347.72 ms, bias select 2208.88 ms
- overview: dense 86.2061 ms/step, hand 1223.2768, bias 949.9961, hand=dense True, bias=dense True, bias=hand True, hand select 547.67 ms, bias select 334.41 ms
- spider_onboarding_design_doc: dense 72.3556 ms/step, hand 1445.2807, bias 1355.3860, hand=dense False, bias=dense False, bias=hand True, hand select 568.60 ms, bias select 488.19 ms
- state_cache_research_note: dense 74.7641 ms/step, hand 1748.3753, bias 1651.1245, hand=dense False, bias=dense False, bias=hand True, hand select 696.86 ms, bias select 622.88 ms
- system-architecture: dense 69.7735 ms/step, hand 941.0337, bias 818.3966, hand=dense False, bias=dense False, bias=hand True, hand select 401.26 ms, bias select 265.82 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
