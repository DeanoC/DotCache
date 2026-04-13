# Qwen3.5 Real Mixed Probe

## Summary

- case count: 8
- hand-tuned avg ms/step: 305.0580
- bias avg ms/step: 304.2026
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 0.375
- hand-tuned direct-M0 query-prep ms/case: 37.9655
- bias direct-M0 query-prep ms/case: 37.7915
- hand-tuned direct-M0 gather ms/case: 52.0710
- bias direct-M0 gather ms/case: 52.0347
- hand-tuned direct-M0 score ms/case: 62.2840
- bias direct-M0 score ms/case: 62.1430
- hand-tuned exact-key M3 score ms/case: 4.0018
- bias exact-key M3 score ms/case: 3.8227
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 92.2269
- bias final-mix ms/case: 76.5842
- hand-tuned final-mix logits ms/case: 16.4929
- bias final-mix logits ms/case: 10.7200
- hand-tuned final-mix softmax ms/case: 75.1912
- bias final-mix softmax ms/case: 65.3258
- hand-tuned final-mix value ms/case: 0.0000
- bias final-mix value ms/case: 0.0000
- hand-tuned executed M0 blocks/case: 4356.00
- bias executed M0 blocks/case: 4356.00
- hand-tuned all-M3 blocks/case: 12.00
- bias all-M3 blocks/case: 12.00
- hand-tuned exact-key M3 blocks/case: 20.00
- bias exact-key M3 blocks/case: 20.00
