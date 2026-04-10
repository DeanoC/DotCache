# Qwen3.5 MPS Paged Attention Snapshot Sweep

## Coverage

- snapshots: `23`
- prompt lengths: `512, 1024, 2048, 4096`
- full-attention layers: `3, 7, 11, 15, 19, 23`
- kv heads: `0, 1`
- pages per snapshot: `9-65`
- total tokens per snapshot: `513-4097`

## Recommendation

Backend and controller are separate axes in this report. `Baseline Backend` means the baseline MPS execution path under the listed controller policy, not the pre-branch system.

| Backend / Controller | Config | Avg step ms | Avg tokens | Avg pages | Pass rate | Max abs err | Max rel err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Experimental Backend / Approx Budget | topk=8|recent=64|sink=64|chunk=8|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.916 | 177.0 | 6.0 | 100.0% | 0.000014 | 0.000133 |
| Baseline Backend / Approx Budget | topk=4|recent=64|sink=64|chunk=2|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.784 | 177.0 | 5.9 | 100.0% | 0.000014 | 0.000133 |

The current winning experimental path (Experimental Backend / Approx Budget) trails Baseline Backend / Approx Budget on the full replay corpus, running at `11.916 ms` versus `11.784 ms`, or `0.989x` of baseline speed. The tradeoff is a smaller active budget: `177.0` average processed tokens for the experimental winner versus `177.0` for the baseline winner.

## Matched Config Speedups

| Config | Baseline ms | Experimental ms | Speedup | Baseline tokens | Experimental tokens |
| --- | --- | --- | --- | --- | --- |
| topk=12|recent=64|sink=64|chunk=8|approx=1|approx_opt=3|early_exit=0|eps=0.0001 | 29.586 | 13.410 | 2.206x | 209.0 | 209.0 |
| topk=12|recent=256|sink=64|chunk=4|approx=1|approx_opt=3|early_exit=0|eps=0.0001 | 50.368 | 23.069 | 2.183x | 397.5 | 397.5 |
| topk=12|recent=256|sink=64|chunk=8|approx=1|approx_opt=3|early_exit=0|eps=0.0001 | 48.533 | 22.376 | 2.169x | 397.5 | 397.5 |
| topk=12|recent=256|sink=64|chunk=2|approx=1|approx_opt=2|early_exit=0|eps=0.0001 | 43.941 | 21.515 | 2.042x | 381.5 | 381.5 |
| topk=12|recent=256|sink=64|chunk=4|approx=1|approx_opt=2|early_exit=0|eps=0.0001 | 44.134 | 21.767 | 2.028x | 381.5 | 381.5 |
| topk=12|recent=128|sink=64|chunk=2|approx=0|approx_opt=0|early_exit=0|eps=0.0001 | 46.405 | 23.747 | 1.954x | 417.0 | 417.0 |
| topk=12|recent=256|sink=64|chunk=8|approx=1|approx_opt=2|early_exit=0|eps=0.0001 | 42.220 | 21.633 | 1.952x | 381.5 | 381.5 |
| topk=12|recent=128|sink=64|chunk=2|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 29.409 | 15.090 | 1.949x | 241.0 | 241.0 |

## Prompt Breakdown

| Prompt | Backend / Controller | Config | Avg step ms | Avg tokens | Avg pages | Max abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | Experimental Backend / Approx Budget | topk=8|recent=64|sink=64|chunk=8|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.929 | 177.0 | 6.0 | 0.000001 |
| 512 | Baseline Backend / Approx Budget | topk=4|recent=64|sink=64|chunk=2|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.791 | 177.0 | 6.0 | 0.000001 |
| 1024 | Experimental Backend / Approx Budget | topk=8|recent=64|sink=64|chunk=8|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.995 | 177.0 | 6.0 | 0.000000 |
| 1024 | Baseline Backend / Approx Budget | topk=4|recent=64|sink=64|chunk=2|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 13.936 | 177.0 | 6.0 | 0.000000 |
| 2048 | Experimental Backend / Approx Budget | topk=8|recent=64|sink=64|chunk=8|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.910 | 177.0 | 6.0 | 0.000014 |
| 2048 | Baseline Backend / Approx Budget | topk=4|recent=64|sink=64|chunk=2|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.661 | 177.0 | 5.9 | 0.000014 |
| 4096 | Experimental Backend / Approx Budget | topk=8|recent=64|sink=64|chunk=8|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.902 | 177.0 | 6.0 | 0.000006 |
| 4096 | Baseline Backend / Approx Budget | topk=4|recent=64|sink=64|chunk=2|approx=1|approx_opt=1|early_exit=0|eps=0.0001 | 11.643 | 177.0 | 5.8 | 0.000004 |

## Experimental Slice Spread

Slowest slices for the winning experimental config:

| Slice | Step ms | Tokens |
| --- | --- | --- |
| prompt4096/layer3/kv0 | 13.309 | 177.0 |
| prompt2048/layer19/kv1 | 12.625 | 177.0 |
| prompt2048/layer7/kv0 | 12.439 | 177.0 |
| prompt2048/layer11/kv1 | 12.370 | 177.0 |
| prompt512/layer19/kv0 | 12.348 | 177.0 |

