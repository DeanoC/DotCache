# Qwen3.5 Persistent Shortlist External Validation

Best priority candidate by worst-case error: `priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2` keeps 0.939 of tokens (1556.8 / 1732.8) with max abs error 0.0325.

Most aggressive candidate: `priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2` keeps 0.939 of tokens (1556.8 / 1732.8) with max abs error 0.0325.

## Config Sweep

| Config | Sel Tokens | Sel Frac | Max Abs | abs<=0.1 | abs<=0.5 | abs<=1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2 | 1556.8 | 0.939 | 0.0325 | 144/144 | 144/144 | 144/144 |
| shortlist_policy_g36 | 1556.8 | 0.939 | 0.0325 | 144/144 | 144/144 | 144/144 |
| shortlist_policy_g36 | 1556.8 | 0.939 | 0.0325 | 144/144 | 144/144 | 144/144 |

## Worst Slices For `priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2`

| Snapshot | Sel Tokens | Max Abs | Max Rel |
| --- | --- | --- | --- |
| codex_algorithmic_alternatives/layer03_kv01_step+07.npz | 2344 | 0.032534 | 4.315 |
| codex_algorithmic_alternatives/layer07_kv00_step+07.npz | 2344 | 0.032147 | 9.345 |
| codex_algorithmic_alternatives/layer03_kv01_step+03.npz | 2340 | 0.021464 | 0.682 |
| codex_algorithmic_alternatives/layer07_kv01_step+07.npz | 2344 | 0.019425 | 1.501 |
| codex_algorithmic_alternatives/layer03_kv01_step+01.npz | 2338 | 0.017880 | 0.617 |
