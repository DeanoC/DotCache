# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 103.6821
- hand-tuned avg ms/step: 3951.7023
- bias avg ms/step: 3784.8608
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.700
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.8326
- hand-tuned score ms/case: 89.0005
- bias score ms/case: 61.2724
- hand-tuned selection ms/case: 1218.6597
- bias selection ms/case: 1189.1981
- hand-tuned optional-selection ms/case: 30.4029
- bias optional-selection ms/case: 28.7068
- hand-tuned diverse-selection ms/case: 5.5198
- bias diverse-selection ms/case: 5.2204
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1155
- bias policy-bias ms/case: 0.1662
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
- hand-tuned executed M3 blocks/case: 5289.60
- bias executed M3 blocks/case: 5289.60
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- decode_pseudocode: dense 179.8079 ms/step, hand 2198.2908, bias 2409.1241, hand=dense True, bias=dense True, bias=hand True, hand select 751.97 ms, bias select 823.76 ms
- layer_revision_tracker: dense 233.8263 ms/step, hand 2982.4632, bias 3088.4455, hand=dense True, bias=dense True, bias=hand True, hand select 1081.03 ms, bias select 1253.68 ms
- local_layer_profiles: dense 67.0368 ms/step, hand 3592.0640, bias 3849.0472, hand=dense False, bias=dense False, bias=hand True, hand select 1373.31 ms, bias select 1453.48 ms
- model_roadmap: dense 71.3668 ms/step, hand 3813.7941, bias 3719.2259, hand=dense True, bias=dense True, bias=hand True, hand select 1395.33 ms, bias select 1399.06 ms
- page_selection_eval: dense 102.1922 ms/step, hand 7709.8739, bias 6283.7313, hand=dense False, bias=dense False, bias=hand True, hand select 2157.54 ms, bias select 1699.60 ms
- performance_journal: dense 93.7812 ms/step, hand 10470.1828, bias 10279.4634, hand=dense False, bias=dense False, bias=hand True, hand select 2328.13 ms, bias select 2215.76 ms
- repo_readme: dense 72.0982 ms/step, hand 3048.3544, bias 2974.9772, hand=dense False, bias=dense False, bias=hand True, hand select 1099.53 ms, bias select 1132.13 ms
- state_cache_roadmap: dense 67.5540 ms/step, hand 1622.2376, bias 1604.2651, hand=dense True, bias=dense True, bias=hand True, hand select 538.50 ms, bias select 573.32 ms
- statecache_showcase: dense 60.0357 ms/step, hand 1655.2924, bias 1592.5653, hand=dense True, bias=dense True, bias=hand True, hand select 549.97 ms, bias select 578.49 ms
- submission_execution_plan: dense 89.1223 ms/step, hand 2424.4699, bias 2047.7632, hand=dense False, bias=dense False, bias=hand True, hand select 911.27 ms, bias select 762.70 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
