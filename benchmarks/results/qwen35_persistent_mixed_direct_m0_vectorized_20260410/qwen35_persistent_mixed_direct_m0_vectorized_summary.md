# Qwen3.5 Persistent Mixed Direct M0 Vectorized Summary

Config:
- model: `Qwen/Qwen3.5-0.8B`
- decode steps: `8`
- key bits: `8`
- mixed strategy: `direct_m0`
- value M0: `off`
- max `K_comp_error`: `0.13`

## Broad 6-prompt set

- dense avg: `98.37 ms/step`
- mixed hand-tuned avg: `2999.63 ms/step`
- mixed bias avg: `2756.57 ms/step`
- bias exact-match vs hand-tuned: `1.0`
- bias faster than hand-tuned on `6/6`

Key per-case costs:
- hand-tuned selection: `1114.94 ms`
- bias selection: `1075.09 ms`
- hand-tuned direct M0 score: `1018.29 ms`
- bias direct M0 score: `93.78 ms`
- hand-tuned exact M3 score: `717.10 ms`
- bias exact M3 score: `103.90 ms`

## Large 10-prompt set

- dense avg: `102.81 ms/step`
- mixed hand-tuned avg: `3645.65 ms/step`
- mixed bias avg: `3097.14 ms/step`
- bias exact-match vs hand-tuned: `1.0`
- bias faster than hand-tuned on `9/10`

Key per-case costs:
- hand-tuned selection: `1350.37 ms`
- bias selection: `1174.62 ms`
- hand-tuned optional selection: `32.26 ms`
- bias optional selection: `29.07 ms`
- hand-tuned direct M0 assembly: `60.43 ms`
- bias direct M0 assembly: `55.33 ms`
- hand-tuned direct M0 score: `1431.34 ms`
- bias direct M0 score: `164.99 ms`
- hand-tuned exact M3 score: `1263.45 ms`
- bias exact M3 score: `165.46 ms`

## Readout

The vectorized direct-M0 work removed selector and assembly as the dominant bottlenecks. The mixed path is now dominated by score kernels, especially on the hand-tuned path. Bias remains the only practical mixed serving mode, but the mixed path is still much slower than the current non-mixed Stage 8 serving baseline (`804.08 ms/step` bias on the checked-in 10-prompt non-mixed run).
