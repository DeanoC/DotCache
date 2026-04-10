# Qwen3.5 Persistent Shortlist External Validation

- unseen prompt families: `codex_algorithmic_alternatives`, `state_cache_research_note`, `spider_onboarding_design_doc`
- evaluated policy: `layer_id + kv_head_id + step_bucket`
- policy bucket found rate: 1.000

## Result

- hand-tuned avg abs: 0.0025901
- runtime-policy avg abs: 0.0025985
- delta avg abs: +0.0000083
- hand-tuned max abs: 0.0325336
- runtime-policy max abs: 0.0325336
- avg selected tokens: 1556.75

## Read

The 36-group runtime policy generalizes structurally on unseen prompt families because it resolves every bucket, but on this external three-family corpus it does not outperform the hand-tuned selector. The average error is slightly worse (`+0.0000083`) and the max error is unchanged, so this is not strong enough evidence yet to replace the hand-tuned policy as the default runtime path.
