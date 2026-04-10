# Qwen3.5 MPS Paged Attention Snapshot Sweep (2026-04-09)

## Coverage

- snapshots: `23`
- prompt lengths: `512, 1024, 2048, 4096`
- full-attention layers: `3, 7, 11, 15, 19, 23`
- kv heads: `0, 1`
- pages per snapshot: `9-65`
- total tokens per snapshot: `513-4097`

## Recommendation

| Engine | Config | Avg step ms | Avg tokens | Avg pages | Pass rate | Max abs err | Max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mps_experimental | topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 1.878 | 449.0 | 8.0 | 100.0% | 0.000015 | 0.000488 |
| torch_mps_baseline | topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 2.054 | 663.3 | 11.3 | 100.0% | 0.000015 | 0.000492 |

The current winning experimental config stays ahead on the full replay corpus, running at `1.878 ms` versus `2.054 ms` for the best baseline, which is a `1.094x` speedup. The tradeoff is a smaller active budget: `449.0` average processed tokens for the experimental winner versus `663.3` for the baseline winner.

## Matched Config Speedups

| Config | Baseline ms | Experimental ms | Speedup | Baseline tokens | Experimental tokens |
| --- | --- | --- | --- | --- | --- |
| topk=4|recent=64|sink=64|chunk=4|early_exit=0|eps=0.0001 | 2.386 | 1.990 | 1.199x | 385.0 | 385.0 |
| topk=4|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 2.217 | 1.942 | 1.142x | 449.0 | 449.0 |
| topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 2.154 | 1.896 | 1.136x | 385.0 | 385.0 |
| topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 2.109 | 1.878 | 1.123x | 449.0 | 449.0 |
| topk=16|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 2.383 | 2.174 | 1.096x | 1055.6 | 1055.6 |
| topk=16|recent=256|sink=64|chunk=8|early_exit=0|eps=0.0001 | 2.367 | 2.256 | 1.049x | 1150.2 | 1150.2 |
| topk=12|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 2.190 | 2.088 | 1.049x | 863.6 | 863.6 |
| topk=8|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 2.171 | 2.077 | 1.046x | 613.2 | 613.2 |

## Prompt Breakdown

| Prompt | Engine | Config | Avg step ms | Avg tokens | Avg pages | Max abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | mps_experimental | topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 1.781 | 449.0 | 8.0 | 0.000001 |
| 512 | torch_mps_baseline | topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 2.047 | 513.0 | 9.0 | 0.000001 |
| 1024 | mps_experimental | topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 1.716 | 449.0 | 8.0 | 0.000000 |
| 1024 | torch_mps_baseline | topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 2.476 | 705.0 | 12.0 | 0.000000 |
| 2048 | mps_experimental | topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 1.915 | 449.0 | 8.0 | 0.000015 |
| 2048 | torch_mps_baseline | topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 2.067 | 705.0 | 12.0 | 0.000015 |
| 4096 | mps_experimental | topk=4|recent=128|sink=64|chunk=8|early_exit=0|eps=0.0001 | 1.919 | 449.0 | 8.0 | 0.000013 |
| 4096 | torch_mps_baseline | topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 1.966 | 705.0 | 12.0 | 0.000011 |

## Experimental Slice Spread

Slowest slices for the winning experimental config:

| Slice | Step ms | Tokens |
| --- | --- | --- |
| prompt2048/layer11/kv1 | 2.371 | 449.0 |
| prompt2048/layer7/kv0 | 2.324 | 449.0 |
| prompt2048/layer15/kv1 | 2.314 | 449.0 |
| prompt4096/layer19/kv1 | 2.238 | 449.0 |
| prompt4096/layer11/kv1 | 2.144 | 449.0 |

