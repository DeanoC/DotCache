# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 74.6850
- hand-tuned avg ms/step: 4496.4806
- bias avg ms/step: 4480.6248
- hand-tuned vs dense exact match rate: 0.167
- bias vs dense exact match rate: 0.167
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.667
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.5557
- hand-tuned score ms/case: 72.1554
- bias score ms/case: 49.8396
- hand-tuned selection ms/case: 18818.9288
- bias selection ms/case: 18742.8179
- hand-tuned policy-bias ms/case: 0.0779
- bias policy-bias ms/case: 0.1160

## Cases

- codex_algorithmic_alternatives: dense 82.8150 ms/step, hand 9192.6923, bias 9237.1037, hand=dense False, bias=dense False, bias=hand True, hand select 42685.56 ms, bias select 42988.16 ms
- dotcache_page_selection_standardized_evaluation: dense 85.2505 ms/step, hand 10361.8276, bias 10282.4679, hand=dense False, bias=dense False, bias=hand True, hand select 48712.32 ms, bias select 48187.94 ms
- overview: dense 67.8622 ms/step, hand 1232.4240, bias 1238.5577, hand=dense True, bias=dense True, bias=hand True, hand select 2524.98 ms, bias select 2497.16 ms
- spider_onboarding_design_doc: dense 70.7827 ms/step, hand 2216.3744, bias 2184.7635, hand=dense False, bias=dense False, bias=hand True, hand select 6875.70 ms, bias select 6814.05 ms
- state_cache_research_note: dense 73.2404 ms/step, hand 2905.3439, bias 2880.0738, hand=dense False, bias=dense False, bias=hand True, hand select 10211.11 ms, bias select 10102.68 ms
- system-architecture: dense 68.1594 ms/step, hand 1070.2214, bias 1060.7824, hand=dense False, bias=dense False, bias=hand True, hand select 1903.89 ms, bias select 1866.92 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
