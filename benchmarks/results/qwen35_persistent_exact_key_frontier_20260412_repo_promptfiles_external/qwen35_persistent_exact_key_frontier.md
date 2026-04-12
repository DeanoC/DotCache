# Qwen3.5 Exact-Key Frontier Study

## Baseline

- case count: 3
- bias avg ms/step: 841.9381
- executed exact-key M3 blocks/case: 8.0000
- candidate layers: [15]
- executed exact-key M3 by layer/case: {"11": 0.0, "15": 8.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}

## Sweeps

- layer `15` at threshold `0.20`:
  - bias avg ms/step: 840.1665
  - delta vs baseline ms/step: -1.7716
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 8.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 8.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
- layer `15` at threshold `0.22`:
  - bias avg ms/step: 1092.8926
  - delta vs baseline ms/step: 250.9545
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 0.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 0.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
- layer `15` at threshold `0.24`:
  - bias avg ms/step: 847.6202
  - delta vs baseline ms/step: 5.6821
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 0.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 0.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
