# Qwen3.5 Real Mixed Probe

## Summary

- case count: 3
- hand-tuned avg ms/step: 269.4204
- bias avg ms/step: 267.1337
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 0.333
- hand-tuned direct-M0 query-prep ms/case: 29.1163
- bias direct-M0 query-prep ms/case: 29.0441
- hand-tuned direct-M0 gather ms/case: 37.7430
- bias direct-M0 gather ms/case: 37.7693
- hand-tuned direct-M0 score ms/case: 68.0484
- bias direct-M0 score ms/case: 67.0181
- hand-tuned exact-key M3 score ms/case: 0.9670
- bias exact-key M3 score ms/case: 0.9547
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 112.0307
- bias final-mix ms/case: 109.8342
- hand-tuned final-mix logits ms/case: 42.4803
- bias final-mix logits ms/case: 42.5996
- hand-tuned final-mix softmax ms/case: 43.6500
- bias final-mix softmax ms/case: 40.8755
- hand-tuned final-mix value ms/case: 25.4212
- bias final-mix value ms/case: 25.8887
- hand-tuned executed M0 blocks/case: 3120.00
- bias executed M0 blocks/case: 3120.00
- hand-tuned all-M3 blocks/case: 0.00
- bias all-M3 blocks/case: 0.00
- hand-tuned exact-key M3 blocks/case: 8.00
- bias exact-key M3 blocks/case: 8.00
