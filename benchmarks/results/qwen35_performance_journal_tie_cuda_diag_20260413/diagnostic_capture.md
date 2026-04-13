# Qwen3.5 `performance_journal` CUDA Tie-Step Diagnostic

## Summary

- target decode step: `5`
- dense generated ids: `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- real mixed generated ids: `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- non-M0 generated ids: `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- dense step-5 argmax: `15`
- real mixed step-5 argmax: `16`
- non-M0 step-5 argmax: `15`
- first real-mixed vs non-M0 output-layer delta above `1e-6`: `3`

## Token 15/16 Readout

- dense logits: `{'15': 20.625, '16': 20.625}`
- dense probs: `{'15': 0.3637197017669678, '16': 0.3637197017669678}`
- real mixed logits: `{'15': 20.625, '16': 20.640625}`
- real mixed probs: `{'15': 0.36158686876296997, '16': 0.367281049489975}`
- non-M0 logits: `{'15': 20.625, '16': 20.625}`
- non-M0 probs: `{'15': 0.3638806641101837, '16': 0.3638806641101837}`

## Interpretation

The first real-mixed vs non-M0 output drift appears at full-attention layer `3`, but the captured direct-M0/final-mix context matches a float32 reference on the same inputs to within `0.00000572`. That points away from the final_mix kernel and toward tiny upstream mixed-path numeric drift before argmax.

## Final-Mix Entries

- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000060`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000060`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000024`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000060`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `0.00000024`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000286`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000072`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000072`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000072`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `0.00000024`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000191`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000107`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000083`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000036`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `0.00000024`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000572`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000072`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000072`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000072`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `0.00000024`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000238`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000191`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `0.00000048`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000238`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000334`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000191`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000191`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000143`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000191`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `0.00000095`, weights-ref max abs delta: `n/a`
