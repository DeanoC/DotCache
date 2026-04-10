# Qwen3.5 Persistent Shortlist Oracle vs Runtime

- Corpus: 144 real long-decode replay snapshots from the 4096-token non-synthetic Qwen corpus.
- Runtime policy table: 108 groups keyed by `layer_id + kv_head_id + prompt_family + step_bucket`.

## Runtime Replay

- Hand-tuned selector avg abs error: 0.0395963
- 108-group runtime policy avg abs error: 0.0394638
- Runtime delta vs hand-tuned: -0.0001324
- Hand-tuned max abs error: 0.2657154
- 108-group runtime policy max abs error: 0.2657154
- Avg selected tokens stayed flat: 2339.75 vs 2339.75
- Policy bucket found rate: 1.000
- Policy applied rate: 1.000

## Offline Decision Quality

- Oracle per-snapshot recommender top-1 accuracy: 0.986
- Oracle per-snapshot chosen-safe rate: 0.764
- Compressed 108-group table top-1 accuracy: 0.861
- Compressed 108-group table chosen-safe rate: 0.757

## Read

- The compressed 108-group table is materially worse than the per-snapshot oracle offline, but it still covers every runtime bucket on this corpus.
- When used as a real runtime selector, it slightly improves average replay error over the current hand-tuned policy without increasing the selected token budget.
- Worst-case error did not move; both runtime policies stayed at 0.2657154.
