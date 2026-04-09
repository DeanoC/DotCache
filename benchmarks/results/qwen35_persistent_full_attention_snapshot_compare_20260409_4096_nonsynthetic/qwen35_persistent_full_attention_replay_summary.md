# Qwen3.5 Persistent Full-Attention Replay Sweep (4096 Real Docs)

Best priority candidate by worst-case error: `priority_recent64_topk128_sink1_explore1` keeps 0.758 of tokens (3106.3 / 4098.3) with max abs error 0.1000.

Most aggressive candidate: `priority_recent1_topk1_sink1_explore1` keeps 0.016 of tokens (66.3 / 4098.3) with max abs error 8.2158.

## Config Sweep

| Config | Sel Tokens | Sel Frac | Max Abs | abs<=0.1 | abs<=0.5 | abs<=1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| full_coverage | 4098.3 | 1.000 | 0.0000 | 108/108 | 108/108 | 108/108 |
| priority_recent64_topk128_sink1_explore1 | 3106.3 | 0.758 | 0.1000 | 107/108 | 108/108 | 108/108 |
| priority_recent32_topk64_sink1_explore1 | 1570.3 | 0.383 | 0.6059 | 85/108 | 107/108 | 108/108 |
| priority_recent16_topk32_sink1_explore1 | 802.3 | 0.196 | 1.6919 | 27/108 | 101/108 | 107/108 |
| priority_recent8_topk16_sink1_explore1 | 418.3 | 0.102 | 3.1297 | 14/108 | 81/108 | 101/108 |
| priority_recent8_topk8_sink1_explore1 | 290.3 | 0.071 | 4.3664 | 9/108 | 69/108 | 92/108 |
| priority_recent4_topk8_sink1_explore1 | 226.3 | 0.055 | 4.6695 | 8/108 | 66/108 | 90/108 |
| priority_recent1_topk8_sink1_explore1 | 178.3 | 0.044 | 5.5548 | 5/108 | 55/108 | 87/108 |
| priority_recent1_topk4_sink1_explore1 | 114.3 | 0.028 | 6.5325 | 2/108 | 44/108 | 76/108 |
| priority_recent1_topk2_sink1_explore1 | 82.3 | 0.020 | 7.5579 | 0/108 | 34/108 | 70/108 |
| priority_recent1_topk1_sink1_explore1 | 66.3 | 0.016 | 8.2158 | 0/108 | 24/108 | 63/108 |

## Worst Slices For `priority_recent64_topk128_sink1_explore1`

| Snapshot | Sel Tokens | Max Abs | Max Rel |
| --- | --- | --- | --- |
| world_interface_agent_loop_paper/layer15_kv01_step+00.npz | 3105 | 0.100005 | 1288.748 |
| readme/layer19_kv00_step+00.npz | 3105 | 0.069988 | 4.130 |
| world_interface_agent_loop_paper/layer19_kv01_step+00.npz | 3105 | 0.058851 | 11.552 |
| readme/layer19_kv01_step+00.npz | 3105 | 0.058301 | 13.801 |
| readme/layer19_kv01_step+01.npz | 3106 | 0.057476 | 6.902 |
