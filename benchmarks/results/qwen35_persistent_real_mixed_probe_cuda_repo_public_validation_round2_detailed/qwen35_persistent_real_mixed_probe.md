# Qwen3.5 Real Mixed Probe

## Summary

- case count: 8
- hand-tuned avg ms/step: 309.1020
- bias avg ms/step: 309.6558
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 0.250
- hand-tuned direct-M0 query-prep ms/case: 37.4712
- bias direct-M0 query-prep ms/case: 37.0483
- hand-tuned direct-M0 gather ms/case: 50.6301
- bias direct-M0 gather ms/case: 50.5859
- hand-tuned direct-M0 score ms/case: 88.0446
- bias direct-M0 score ms/case: 87.1950
- hand-tuned exact-key M3 score ms/case: 3.8758
- bias exact-key M3 score ms/case: 3.6457
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 145.3907
- bias final-mix ms/case: 144.0897
- hand-tuned final-mix logits ms/case: 57.1929
- bias final-mix logits ms/case: 57.1749
- hand-tuned final-mix softmax ms/case: 53.0934
- bias final-mix softmax ms/case: 52.0355
- hand-tuned final-mix value ms/case: 34.3407
- bias final-mix value ms/case: 34.1246
- hand-tuned executed M0 blocks/case: 4356.00
- bias executed M0 blocks/case: 4356.00
- hand-tuned all-M3 blocks/case: 12.00
- bias all-M3 blocks/case: 12.00
- hand-tuned exact-key M3 blocks/case: 20.00
- bias exact-key M3 blocks/case: 20.00
