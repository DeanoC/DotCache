# Qwen3.5 Real Mixed Probe

## Summary

- case count: 1
- hand-tuned avg ms/step: 409.5302
- bias avg ms/step: 397.1591
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 1.000
- hand-tuned direct-M0 query-prep ms/case: 50.5518
- bias direct-M0 query-prep ms/case: 50.4240
- hand-tuned direct-M0 gather ms/case: 68.0677
- bias direct-M0 gather ms/case: 68.4685
- hand-tuned direct-M0 score ms/case: 122.4482
- bias direct-M0 score ms/case: 118.1439
- hand-tuned exact-key M3 score ms/case: 1.0300
- bias exact-key M3 score ms/case: 0.9976
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 203.9405
- bias final-mix ms/case: 194.6770
- hand-tuned final-mix logits ms/case: 77.8407
- bias final-mix logits ms/case: 77.5112
- hand-tuned final-mix softmax ms/case: 78.9575
- bias final-mix softmax ms/case: 70.4535
- hand-tuned final-mix value ms/case: 46.2185
- bias final-mix value ms/case: 45.7871
- hand-tuned executed M0 blocks/case: 6192.00
- bias executed M0 blocks/case: 6192.00
- hand-tuned all-M3 blocks/case: 0.00
- bias all-M3 blocks/case: 0.00
- hand-tuned exact-key M3 blocks/case: 8.00
- bias exact-key M3 blocks/case: 8.00
