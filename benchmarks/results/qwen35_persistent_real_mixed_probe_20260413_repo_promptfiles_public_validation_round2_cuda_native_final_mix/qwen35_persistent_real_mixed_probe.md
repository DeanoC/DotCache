# Qwen3.5 Real Mixed Probe

## Summary

- case count: 8
- hand-tuned avg ms/step: 308.9416
- bias avg ms/step: 308.7841
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 0.250
- hand-tuned direct-M0 query-prep ms/case: 37.4626
- bias direct-M0 query-prep ms/case: 37.2530
- hand-tuned direct-M0 gather ms/case: 50.6015
- bias direct-M0 gather ms/case: 50.1982
- hand-tuned direct-M0 score ms/case: 87.7123
- bias direct-M0 score ms/case: 86.6854
- hand-tuned exact-key M3 score ms/case: 3.7896
- bias exact-key M3 score ms/case: 3.6964
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 145.3938
- bias final-mix ms/case: 143.2065
- hand-tuned final-mix logits ms/case: 57.5082
- bias final-mix logits ms/case: 56.8857
- hand-tuned final-mix softmax ms/case: 53.1705
- bias final-mix softmax ms/case: 51.8943
- hand-tuned final-mix value ms/case: 34.0080
- bias final-mix value ms/case: 33.7172
- hand-tuned executed M0 blocks/case: 4356.00
- bias executed M0 blocks/case: 4356.00
- hand-tuned all-M3 blocks/case: 12.00
- bias all-M3 blocks/case: 12.00
- hand-tuned exact-key M3 blocks/case: 20.00
- bias exact-key M3 blocks/case: 20.00
