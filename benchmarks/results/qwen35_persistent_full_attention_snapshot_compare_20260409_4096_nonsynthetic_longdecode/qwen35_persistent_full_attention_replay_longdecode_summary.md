# Qwen3.5 Persistent Full-Attention Replay Long-Decode Sweep

Best priority candidate by worst-case error: `priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2` keeps 0.571 of tokens (2339.8 / 4099.8) with max abs error 0.2756.

Most aggressive candidate: `priority_recent64_topk128_sink1_explore1` keeps 0.571 of tokens (2339.8 / 4099.8) with max abs error 0.2756.

## Config Sweep

| Config | Sel Tokens | Sel Frac | Max Abs | abs<=0.1 | abs<=0.5 | abs<=1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| priority_recent64_topk128_sink1_explore1 | 2339.8 | 0.571 | 0.2756 | 132/144 | 144/144 | 144/144 |
| priority_recent64_topk128_sink1_explore1_histema | 2339.8 | 0.571 | 0.2756 | 132/144 | 144/144 | 144/144 |
| priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate | 2339.8 | 0.571 | 0.2756 | 133/144 | 144/144 | 144/144 |
| priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2 | 2339.8 | 0.571 | 0.2756 | 133/144 | 144/144 | 144/144 |

## Worst Slices For `priority_recent64_topk128_sink1_explore1_histema_div0.5_r4_histgate_h1to2`

| Snapshot | Sel Tokens | Max Abs | Max Rel |
| --- | --- | --- | --- |
| world_interface_agent_loop_paper/layer15_kv01_step+00.npz | 2337 | 0.275561 | 3678.517 |
| readme/layer19_kv01_step+00.npz | 2337 | 0.164371 | 16.666 |
| world_interface_agent_loop_paper/layer15_kv01_step+01.npz | 2338 | 0.160187 | 146.791 |
| readme/layer19_kv01_step+01.npz | 2338 | 0.156448 | 8.842 |
| readme/layer19_kv00_step+00.npz | 2337 | 0.148251 | 7.440 |
