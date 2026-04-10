# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 3
- dense avg ms/step: 74.7447
- hand-tuned avg ms/step: 4933.4629
- bias avg ms/step: 4867.9196
- hand-tuned vs dense exact match rate: 0.000
- bias vs dense exact match rate: 0.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.7339
- hand-tuned score ms/case: 70.5276
- bias score ms/case: 50.4537
- hand-tuned selection ms/case: 20628.1010
- bias selection ms/case: 20300.9099
- hand-tuned policy-bias ms/case: 0.0892
- bias policy-bias ms/case: 0.1324

## Cases

- codex_algorithmic_alternatives: dense 82.5676 ms/step, hand 9549.7865, bias 9396.9237, hand=dense False, bias=dense False, bias=hand True, hand select 44577.83 ms, bias select 43537.13 ms
- spider_onboarding_design_doc: dense 70.8451 ms/step, hand 2272.5296, bias 2242.7196, hand=dense False, bias=dense False, bias=hand True, hand select 6947.37 ms, bias select 7004.09 ms
- state_cache_research_note: dense 70.8213 ms/step, hand 2978.0725, bias 2964.1157, hand=dense False, bias=dense False, bias=hand True, hand select 10359.10 ms, bias select 10361.51 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
