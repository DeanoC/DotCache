# Qwen3.5 MPS Paged Attention Snapshot Sweep (Robust Rewrite)

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
| mps_experimental | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.847 | 225.0 | 7.8 | 100.0% | 0.000014 | 0.000269 |
| torch_mps_baseline | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.511 | 225.0 | 7.8 | 100.0% | 0.000014 | 0.000269 |

The current winning experimental config trails the best baseline on the full replay corpus, running at `15.847 ms` versus `15.511 ms`, or `0.979x` of baseline speed. The tradeoff is a smaller active budget: `225.0` average processed tokens for the experimental winner versus `225.0` for the baseline winner.

## Matched Config Speedups

| Config | Baseline ms | Experimental ms | Speedup | Baseline tokens | Experimental tokens |
| --- | --- | --- | --- | --- | --- |
| topk=8|recent=64|sink=64|chunk=4|early_exit=0|eps=0.0001 | 26.498 | 18.421 | 1.438x | 289.0 | 289.0 |
| topk=8|recent=64|sink=64|chunk=2|early_exit=0|eps=0.0001 | 26.818 | 19.236 | 1.394x | 289.0 | 289.0 |
| topk=4|recent=256|sink=64|chunk=2|early_exit=0|eps=0.0001 | 34.221 | 24.683 | 1.386x | 413.5 | 413.5 |
| topk=12|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 30.038 | 23.172 | 1.296x | 353.0 | 353.0 |
| topk=4|recent=256|sink=64|chunk=4|early_exit=0|eps=0.0001 | 31.495 | 24.666 | 1.277x | 413.5 | 413.5 |
| topk=12|recent=256|sink=64|chunk=8|early_exit=0|eps=0.0001 | 41.866 | 34.323 | 1.220x | 538.0 | 538.0 |
| topk=8|recent=128|sink=64|chunk=4|early_exit=0|eps=0.0001 | 36.263 | 30.266 | 1.198x | 353.0 | 353.0 |
| topk=12|recent=128|sink=64|chunk=2|early_exit=0|eps=0.0001 | 34.769 | 29.559 | 1.176x | 417.0 | 417.0 |

## Prompt Breakdown

| Prompt | Engine | Config | Avg step ms | Avg tokens | Avg pages | Max abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | mps_experimental | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.258 | 225.0 | 7.4 | 0.000001 |
| 512 | torch_mps_baseline | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.418 | 225.0 | 7.4 | 0.000001 |
| 1024 | mps_experimental | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 14.880 | 225.0 | 8.0 | 0.000000 |
| 1024 | torch_mps_baseline | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 16.535 | 225.0 | 8.0 | 0.000000 |
| 2048 | mps_experimental | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 16.640 | 225.0 | 7.8 | 0.000014 |
| 2048 | torch_mps_baseline | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.561 | 225.0 | 7.8 | 0.000014 |
| 4096 | mps_experimental | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.045 | 225.0 | 8.0 | 0.000010 |
| 4096 | torch_mps_baseline | topk=4|recent=64|sink=64|chunk=8|early_exit=0|eps=0.0001 | 15.327 | 225.0 | 8.0 | 0.000010 |

## Experimental Slice Spread

Slowest slices for the winning experimental config:

| Slice | Step ms | Tokens |
| --- | --- | --- |
| prompt2048/layer19/kv0 | 29.694 | 225.0 |
| prompt2048/layer23/kv0 | 22.985 | 225.0 |
| prompt512/layer11/kv0 | 17.310 | 225.0 |
| prompt4096/layer3/kv1 | 15.514 | 225.0 |
| prompt4096/layer19/kv0 | 15.232 | 225.0 |

