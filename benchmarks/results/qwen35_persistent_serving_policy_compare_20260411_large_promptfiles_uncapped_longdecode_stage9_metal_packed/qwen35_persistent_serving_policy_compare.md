# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 124.5226
- hand-tuned avg ms/step: 2779.6269
- bias avg ms/step: 2509.1094
- hand-tuned vs dense exact match rate: 0.800
- bias vs dense exact match rate: 0.800
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.900
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 2.7374
- hand-tuned score ms/case: 120.9658
- bias score ms/case: 44.6892
- hand-tuned selection ms/case: 315.8099
- bias selection ms/case: 236.9735
- hand-tuned optional-selection ms/case: 7.4454
- bias optional-selection ms/case: 7.4104
- hand-tuned diverse-selection ms/case: 1.9870
- bias diverse-selection ms/case: 1.8919
- hand-tuned compression-selection ms/case: 1.0698
- bias compression-selection ms/case: 1.0896
- hand-tuned policy-bias ms/case: 0.0258
- bias policy-bias ms/case: 0.0306
- hand-tuned direct-M0 assembly ms/case: 11.4913
- bias direct-M0 assembly ms/case: 11.1529
- hand-tuned direct-M0 score ms/case: 225.0101
- bias direct-M0 score ms/case: 33.2994
- hand-tuned exact-M3 score ms/case: 158.1099
- bias exact-M3 score ms/case: 31.4529
- hand-tuned final-mix ms/case: 29.4475
- bias final-mix ms/case: 7.5639

## Cases

- AAE_DotCache_Robust_Rewrite: dense 177.7454 ms/step, hand 4462.6945, bias 4356.8186, hand=dense True, bias=dense True, bias=hand True, hand select 429.15 ms, bias select 407.65 ms
- Minimal Agent Loop: dense 126.3206 ms/step, hand 3858.4371, bias 3629.5339, hand=dense True, bias=dense True, bias=hand True, hand select 397.68 ms, bias select 349.74 ms
- auth-and-boundaries: dense 77.3843 ms/step, hand 857.7250, bias 681.8665, hand=dense False, bias=dense False, bias=hand True, hand select 108.57 ms, bias select 67.85 ms
- codex_algorithmic_alternatives: dense 146.1262 ms/step, hand 4495.6325, bias 3966.8804, hand=dense True, bias=dense True, bias=hand True, hand select 427.04 ms, bias select 375.32 ms
- external-worker-flow: dense 93.1891 ms/step, hand 1046.1482, bias 892.1162, hand=dense True, bias=dense True, bias=hand True, hand select 128.71 ms, bias select 92.56 ms
- model_roadmap: dense 161.7873 ms/step, hand 6336.9376, bias 6360.9575, hand=dense False, bias=dense False, bias=hand True, hand select 632.25 ms, bias select 545.90 ms
- overview: dense 111.5110 ms/step, hand 1400.8629, bias 967.2901, hand=dense True, bias=dense True, bias=hand True, hand select 272.23 ms, bias select 93.56 ms
- spider_onboarding_design_doc: dense 127.9210 ms/step, hand 1791.6165, bias 1602.7171, hand=dense True, bias=dense True, bias=hand True, hand select 229.94 ms, bias select 172.48 ms
- state_cache_research_note: dense 126.6538 ms/step, hand 2152.3714, bias 1721.4419, hand=dense True, bias=dense True, bias=hand True, hand select 258.57 ms, bias select 176.14 ms
- system-architecture: dense 96.5874 ms/step, hand 1393.8432, bias 911.4719, hand=dense True, bias=dense True, bias=hand True, hand select 273.96 ms, bias select 88.54 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
