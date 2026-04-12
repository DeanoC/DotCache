# Qwen3.5 Exact-Key Frontier Study

## Baseline

- case count: 10
- bias avg ms/step: 1656.6447
- executed exact-key M3 blocks/case: 8.0000
- candidate layers: [15]
- executed exact-key M3 by layer/case: {"11": 0.0, "15": 8.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}

## Sweeps

- layer `15` at threshold `0.20`:
  - bias avg ms/step: 1515.2186
  - delta vs baseline ms/step: -141.4261
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 8.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 8.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
- layer `15` at threshold `0.22`:
  - bias avg ms/step: 2305.6823
  - delta vs baseline ms/step: 649.0376
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 0.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 0.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
- layer `15` at threshold `0.24`:
  - bias avg ms/step: 1661.5906
  - delta vs baseline ms/step: 4.9459
  - bias vs baseline exact match rate: 1.000
  - executed exact-key M3 blocks/case: 0.0000
  - executed exact-key M3 by layer/case: {"11": 0.0, "15": 0.0, "19": 0.0, "23": 0.0, "3": 0.0, "7": 0.0}
