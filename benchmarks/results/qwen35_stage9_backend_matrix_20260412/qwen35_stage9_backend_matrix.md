# Qwen3.5 Stage 9 Backend Matrix

## Winners

| Corpus | MPS winner | CUDA winner |
| --- | --- | --- |
| large | real_mixed `1407.44` | real_mixed `399.74` |
| broad | real_mixed `1627.72` | real_mixed `460.70` |
| external | real_mixed `843.77` | non_m0 `192.54` |

## Matrix

| Corpus | Backend | Policy | Bias ms/step | Hand ms/step | Exact-match | Bias beats hand |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| large | mps | real_mixed | `1407.44` | `2590.71` | `1.000` | `1.000` |
| large | mps | non_m0 | `3784.86` | `3951.70` | `1.000` | `0.700` |
| large | mps | conservative | `2054.42` | `2108.73` | `1.000` | `1.000` |
| large | cuda | real_mixed | `399.74` | `398.29` | `1.000` | `0.100` |
| large | cuda | non_m0 | `468.69` | `467.89` | `1.000` | `0.200` |
| large | cuda | conservative | `652.91` | `652.15` | `1.000` | `0.200` |
| broad | mps | real_mixed | `1627.72` | `2583.39` | `1.000` | `1.000` |
| broad | mps | non_m0 | `4234.40` | `4289.79` | `1.000` | `0.833` |
| broad | mps | conservative | `2828.78` | `2484.60` | `1.000` | `0.667` |
| broad | cuda | real_mixed | `460.70` | `461.39` | `1.000` | `0.333` |
| broad | cuda | non_m0 | `632.12` | `632.51` | `1.000` | `0.333` |
| broad | cuda | conservative | `792.60` | `791.37` | `1.000` | `0.167` |
| external | mps | real_mixed | `843.77` | `1290.56` | `1.000` | `1.000` |
| external | mps | non_m0 | `1700.51` | `1609.82` | `1.000` | `0.333` |
| external | mps | conservative | `1340.05` | `1176.19` | `1.000` | `0.667` |
| external | cuda | real_mixed | `252.42` | `254.46` | `1.000` | `0.333` |
| external | cuda | non_m0 | `192.54` | `193.63` | `1.000` | `0.333` |
| external | cuda | conservative | `380.85` | `381.95` | `1.000` | `0.333` |
