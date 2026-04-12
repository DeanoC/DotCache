# Qwen3.5 Exact-Key Frontier Study

## Baseline

- case count: 6
- bias avg ms/step: 1864.5783
- executed exact-key M3 blocks/case: 8.0000
- candidate layers: [15]
- executed exact-key M3 by layer/case: {"11": 0.0, "15": 8.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}

## Sweeps

- layer `15` at threshold `0.20`:
  - bias avg ms/step: 1772.4438
  - delta vs baseline ms/step: -92.1345
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 8.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 8.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
- layer `15` at threshold `0.22`:
  - bias avg ms/step: 2205.6307
  - delta vs baseline ms/step: 341.0524
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 0.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 0.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
- layer `15` at threshold `0.24`:
  - bias avg ms/step: 1693.0541
  - delta vs baseline ms/step: -171.5242
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 0.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 0.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
