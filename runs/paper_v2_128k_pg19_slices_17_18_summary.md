# Paper v2 128K PG-19 Slice Results: 17-18

Generated on branch `port-to-paper-20260424` with the paper-quality
`full-bounded` cache mode.

## Slice Summary

| Slice | Source dir | Wall | Dense ppl | Certified ppl | Ratio | Delta |
|---:|---|---:|---:|---:|---:|---:|
| 17 | `runs/paper_v2_distributed_128k_pg19_slices_17_18` | 358.2m | 6.7934 | 6.7945 | 1.000162 | +0.0011 |
| 18 | `runs/paper_v2_distributed_128k_pg19_slices_17_18` | 358.8m | 6.6040 | 6.6034 | 0.999914 | -0.0006 |

## Aggregate Checks

- PG-19 ratio mean for slices 17-18: `1.000038`.
- PG-19 delta mean for slices 17-18: `+0.00027` ppl.
- Average slice time: `358.5m`.
- Total machine time: `717.1m`.

## Notes

- Both completed benchmark jobs in slice manifests have `exit_code: 0`.
- Context: `131072`.
- FP16 key/value cache blocks: `9216` / `9216`.
- Hard-stop counters spot-checked as zero: score consistency violations, boundary check fired steps, and Rung 4 fired steps.
- Source directory: `runs/paper_v2_distributed_128k_pg19_slices_17_18`.
