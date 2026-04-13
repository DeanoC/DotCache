# Qwen3.5 Real Mixed Probe

## Summary

- case count: 2
- hand-tuned avg ms/step: 410.6975
- bias avg ms/step: 407.8449
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 0.500
- hand-tuned direct-M0 query-prep ms/case: 53.8915
- bias direct-M0 query-prep ms/case: 50.7427
- hand-tuned direct-M0 gather ms/case: 73.9303
- bias direct-M0 gather ms/case: 66.7140
- hand-tuned direct-M0 score ms/case: 126.9804
- bias direct-M0 score ms/case: 123.9103
- hand-tuned exact-key M3 score ms/case: 7.2950
- bias exact-key M3 score ms/case: 6.8432
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 207.3056
- bias final-mix ms/case: 195.3404
- hand-tuned final-mix logits ms/case: 81.8643
- bias final-mix logits ms/case: 77.1567
- hand-tuned final-mix softmax ms/case: 78.0700
- bias final-mix softmax ms/case: 72.0100
- hand-tuned final-mix value ms/case: 46.4192
- bias final-mix value ms/case: 45.2426
- hand-tuned executed M0 blocks/case: 5112.00
- bias executed M0 blocks/case: 5112.00
- hand-tuned all-M3 blocks/case: 24.00
- bias all-M3 blocks/case: 24.00
- hand-tuned exact-key M3 blocks/case: 32.00
- bias exact-key M3 blocks/case: 32.00
