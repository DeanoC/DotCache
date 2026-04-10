# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 78.0493
- hand-tuned avg ms/step: 5303.1848
- bias avg ms/step: 5225.9222
- hand-tuned vs dense exact match rate: 0.300
- bias vs dense exact match rate: 0.300
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.900
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.8527
- hand-tuned score ms/case: 85.0376
- bias score ms/case: 51.0511
- hand-tuned selection ms/case: 22787.1563
- bias selection ms/case: 22573.7089
- hand-tuned policy-bias ms/case: 0.0926
- bias policy-bias ms/case: 0.1186

## Cases

- AAE_DotCache_Robust_Rewrite: dense 89.5343 ms/step, hand 10766.4152, bias 10645.3100, hand=dense True, bias=dense True, bias=hand True, hand select 50839.77 ms, bias select 50304.02 ms
- Minimal Agent Loop: dense 85.0071 ms/step, hand 8655.3678, bias 8555.0228, hand=dense True, bias=dense True, bias=hand True, hand select 39799.71 ms, bias select 39450.93 ms
- auth-and-boundaries: dense 69.4251 ms/step, hand 738.0426, bias 721.5911, hand=dense False, bias=dense False, bias=hand True, hand select 636.09 ms, bias select 601.42 ms
- codex_algorithmic_alternatives: dense 89.0614 ms/step, hand 9583.5556, bias 9379.1517, hand=dense False, bias=dense False, bias=hand True, hand select 44271.46 ms, bias select 43491.65 ms
- external-worker-flow: dense 67.9944 ms/step, hand 1026.0364, bias 991.1507, hand=dense False, bias=dense False, bias=hand True, hand select 1597.48 ms, bias select 1563.28 ms
- model_roadmap: dense 90.2850 ms/step, hand 14426.8150, bias 14470.8058, hand=dense False, bias=dense False, bias=hand True, hand select 68800.96 ms, bias select 68802.14 ms
- overview: dense 71.5329 ms/step, hand 1253.3922, bias 1241.1354, hand=dense True, bias=dense True, bias=hand True, hand select 2519.23 ms, bias select 2530.60 ms
- spider_onboarding_design_doc: dense 71.4560 ms/step, hand 2431.2195, bias 2228.3495, hand=dense False, bias=dense False, bias=hand True, hand select 7031.52 ms, bias select 6944.53 ms
- state_cache_research_note: dense 73.3954 ms/step, hand 3017.6395, bias 2940.4573, hand=dense False, bias=dense False, bias=hand True, hand select 10395.31 ms, bias select 10148.95 ms
- system-architecture: dense 72.8016 ms/step, hand 1133.3646, bias 1086.2477, hand=dense False, bias=dense False, bias=hand True, hand select 1980.05 ms, bias select 1899.56 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
