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

The default same-tree run reproduces the real-mixed `15`/`16` flip and localizes the first real-mixed vs non-M0 output drift to full-attention layer `3`. This pass did not include per-head final-mix input capture, so it does not by itself implicate the final_mix helper.

## Final-Mix Entries

- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `3` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `7` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `11` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `15` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `19` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
- layer `23` `direct_m0_stream_stats` context-ref max abs delta: `n/a`, weights-ref max abs delta: `n/a`
