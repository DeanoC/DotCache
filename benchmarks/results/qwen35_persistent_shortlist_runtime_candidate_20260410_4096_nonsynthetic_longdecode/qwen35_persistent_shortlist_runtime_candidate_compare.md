# Qwen3.5 Persistent Shortlist Runtime Candidate

Best priority candidate by worst-case error: `shortlist_policy_g36` keeps 0.571 of tokens (2339.8 / 4099.8) with max abs error 0.2657.

Most aggressive candidate: `priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2` keeps 0.571 of tokens (2339.8 / 4099.8) with max abs error 0.2657.

## Config Sweep

| Config | Sel Tokens | Sel Frac | Max Abs | abs<=0.1 | abs<=0.5 | abs<=1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2 | 2339.8 | 0.571 | 0.2657 | 133/144 | 144/144 | 144/144 |
| shortlist_policy_g36 | 2339.8 | 0.571 | 0.2657 | 133/144 | 144/144 | 144/144 |

## Worst Slices For `shortlist_policy_g36`

| Snapshot | Sel Tokens | Max Abs | Max Rel |
| --- | --- | --- | --- |
| world_interface_agent_loop_paper/layer15_kv01_step+00.npz | 2337 | 0.265715 | 2709.661 |
| world_interface_agent_loop_paper/layer15_kv01_step+01.npz | 2338 | 0.160187 | 146.791 |
| readme/layer19_kv01_step+01.npz | 2338 | 0.156448 | 8.842 |
| readme/layer19_kv01_step+00.npz | 2337 | 0.152097 | 18.468 |
| readme/layer19_kv00_step+00.npz | 2337 | 0.151809 | 6.420 |
