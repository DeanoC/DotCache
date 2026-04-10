# Qwen3.5 Persistent Shortlist External Validation (Four-Way)

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

## Disagreement Analysis

- hard override regressions vs hand-tuned: 3
- assist regressions vs hand-tuned: 0
- worst hard-override regression: codex_algorithmic_alternatives layer 3 kv 0 step 3 (0.011306 -> 0.013176, delta 0.001869)

## Read

The hard shortlist override regresses only a few unseen-family slices, and all of those regressions disappear in assist mode.
That points to the policy table flattening useful hand-tuned heuristics, especially diversity and far-anchor behavior, rather than improving the underlying budget shape.
Assist mode preserves the hand-tuned heuristics and matches the hand-tuned external result while still resolving policy buckets on every snapshot.
