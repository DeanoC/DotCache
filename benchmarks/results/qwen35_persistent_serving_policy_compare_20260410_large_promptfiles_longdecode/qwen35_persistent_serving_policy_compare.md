# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 57.3707
- hand-tuned avg ms/step: 829.1330
- bias avg ms/step: 804.0753
- hand-tuned vs dense exact match rate: 0.300
- bias vs dense exact match rate: 0.300
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 8.0146
- hand-tuned score ms/case: 57.8695
- bias score ms/case: 49.7635
- hand-tuned selection ms/case: 707.5874
- bias selection ms/case: 699.7642
- hand-tuned optional-selection ms/case: 388.1765
- bias optional-selection ms/case: 387.0520
- hand-tuned diverse-selection ms/case: 56.4468
- bias diverse-selection ms/case: 55.6299
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0665
- bias policy-bias ms/case: 0.0909

## Cases

- AAE_DotCache_Robust_Rewrite: dense 62.3172 ms/step, hand 865.7311, bias 818.2177, hand=dense False, bias=dense False, bias=hand True, hand select 762.05 ms, bias select 728.87 ms
- Minimal Agent Loop: dense 55.4635 ms/step, hand 837.0941, bias 826.5331, hand=dense False, bias=dense False, bias=hand True, hand select 725.30 ms, bias select 727.19 ms
- auth-and-boundaries: dense 68.0923 ms/step, hand 725.2344, bias 681.1001, hand=dense False, bias=dense False, bias=hand True, hand select 514.12 ms, bias select 475.85 ms
- codex_algorithmic_alternatives: dense 58.6842 ms/step, hand 836.5586, bias 816.7460, hand=dense False, bias=dense False, bias=hand True, hand select 724.06 ms, bias select 723.73 ms
- external-worker-flow: dense 57.2932 ms/step, hand 835.5810, bias 816.3529, hand=dense False, bias=dense False, bias=hand True, hand select 729.48 ms, bias select 724.26 ms
- model_roadmap: dense 56.6998 ms/step, hand 840.3675, bias 816.4833, hand=dense True, bias=dense True, bias=hand True, hand select 725.24 ms, bias select 723.39 ms
- overview: dense 53.0737 ms/step, hand 837.8129, bias 821.6653, hand=dense False, bias=dense False, bias=hand True, hand select 725.05 ms, bias select 736.76 ms
- spider_onboarding_design_doc: dense 56.6280 ms/step, hand 836.5239, bias 817.8557, hand=dense False, bias=dense False, bias=hand True, hand select 724.66 ms, bias select 724.90 ms
- state_cache_research_note: dense 49.5229 ms/step, hand 842.4403, bias 810.0763, hand=dense True, bias=dense True, bias=hand True, hand select 719.49 ms, bias select 711.21 ms
- system-architecture: dense 55.9319 ms/step, hand 833.9866, bias 815.7230, hand=dense True, bias=dense True, bias=hand True, hand select 726.44 ms, bias select 721.49 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
