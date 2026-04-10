# Qwen3.5 Persistent Shortlist External Validation (Five-Way)

## Results

- hand-tuned avg abs: 0.0025901
- hand-tuned max abs: 0.0325336
- hand-tuned avg selected tokens: 1556.75
- hand-tuned policy applied rate: 0.000
- hand-tuned bucket found rate: 0.000
- policy all-steps avg abs: 0.0025985
- policy all-steps max abs: 0.0325336
- policy all-steps avg selected tokens: 1556.75
- policy all-steps policy applied rate: 1.000
- policy all-steps bucket found rate: 1.000
- policy mid/late-only avg abs: 0.0025985
- policy mid/late-only max abs: 0.0325336
- policy mid/late-only avg selected tokens: 1556.75
- policy mid/late-only policy applied rate: 0.750
- policy mid/late-only bucket found rate: 0.750
- policy assist avg abs: 0.0025901
- policy assist max abs: 0.0325336
- policy assist avg selected tokens: 1556.75
- policy assist policy applied rate: 1.000
- policy assist bucket found rate: 1.000
- policy bias avg abs: 0.0018027
- policy bias max abs: 0.0269186
- policy bias avg selected tokens: 1556.75
- policy bias policy applied rate: 1.000
- policy bias bucket found rate: 1.000

## Read

Bias-mode shortlist guidance is the first policy-assisted variant that clearly beats the hand-tuned selector on the unseen-family corpus at the same token budget.
Hard override regressed a few slices, assist matched hand-tuned, and bias improved both average and worst-case error while still resolving policy buckets on every snapshot.
The small bias sweep also stayed stable across 0.02 to 0.10, with 0.10 edging out the lower settings by a small margin, so bias is now the preferred experimental mode.
