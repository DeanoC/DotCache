# Qwen3.5 Real Mixed Probe

## Summary

- case count: 8
- hand-tuned avg ms/step: 305.8164
- bias avg ms/step: 306.5711
- bias vs hand exact match rate: 1.000
- bias beats hand-tuned latency rate: 0.250
- hand-tuned direct-M0 query-prep ms/case: 37.0281
- bias direct-M0 query-prep ms/case: 36.9575
- hand-tuned direct-M0 gather ms/case: 50.3315
- bias direct-M0 gather ms/case: 50.1207
- hand-tuned direct-M0 score ms/case: 60.8588
- bias direct-M0 score ms/case: 60.3289
- hand-tuned exact-key M3 score ms/case: 3.8584
- bias exact-key M3 score ms/case: 3.6932
- hand-tuned aux exact-M3 score ms/case: 0.0000
- bias aux exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 146.8130
- bias final-mix ms/case: 144.9095
- hand-tuned final-mix logits ms/case: 57.5298
- bias final-mix logits ms/case: 57.3360
- hand-tuned final-mix softmax ms/case: 53.0674
- bias final-mix softmax ms/case: 51.9817
- hand-tuned final-mix value ms/case: 35.5353
- bias final-mix value ms/case: 34.9159
- hand-tuned executed M0 blocks/case: 4356.00
- bias executed M0 blocks/case: 4356.00
- hand-tuned all-M3 blocks/case: 12.00
- bias all-M3 blocks/case: 12.00
- hand-tuned exact-key M3 blocks/case: 20.00
- bias exact-key M3 blocks/case: 20.00
