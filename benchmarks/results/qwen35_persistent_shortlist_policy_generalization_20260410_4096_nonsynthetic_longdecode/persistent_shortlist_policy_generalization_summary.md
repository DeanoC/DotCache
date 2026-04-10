# Persistent Shortlist Policy Generalization

- prompt families: performance_journal, readme, world_interface_agent_loop_paper
- abs threshold: 0.0500

## layer_id, kv_head_id, step_bucket

- policy groups (mean across held-out runs): 36.0
- top-1 accuracy: 0.625
- chosen-safe rate: 0.743
- avg selected tokens: 2339.8
- oracle avg selected tokens: 2339.8
- fallback rate: 0.000
- missing bucket rate: 0.000

| Holdout family | Groups | Top-1 | Safe rate | Fallback | Missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| performance_journal | 36 | 0.667 | 0.979 | 0.000 | 0.000 |
| readme | 36 | 0.583 | 0.812 | 0.000 | 0.000 |
| world_interface_agent_loop_paper | 36 | 0.625 | 0.438 | 0.000 | 0.000 |

## layer_id, kv_head_id, prompt_family, step_bucket

- policy groups (mean across held-out runs): 72.0
- top-1 accuracy: 1.000
- chosen-safe rate: 0.764
- avg selected tokens: 2339.8
- oracle avg selected tokens: 2339.8
- fallback rate: 1.000
- missing bucket rate: 1.000

Note: non-zero missing-bucket rate means some or all held-out results are driven by fallback, not by policy matches.

| Holdout family | Groups | Top-1 | Safe rate | Fallback | Missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| performance_journal | 72 | 1.000 | 0.979 | 1.000 | 1.000 |
| readme | 72 | 1.000 | 0.833 | 1.000 | 1.000 |
| world_interface_agent_loop_paper | 72 | 1.000 | 0.479 | 1.000 | 1.000 |
