# Runtime Page Selector Artifact

- classes: 2
- training examples: 352
- runtime feature count: 13
- artifact path: benchmarks/results/qwen35_selector_qwen35_0p8b_local_20260406/serving_selector_artifact_family_reasoning_holdout/linear_selector_model.json
- class balance exponent: 0.50
- safe bytes weight: 1.00
- reference candidate: M3/affine/4/float16

## Features

- `stage_decode`
- `kind_key`
- `layer_fraction`
- `kv_head_fraction`
- `log_token_start`
- `log_token_age`
- `token_count`
- `head_dim`
- `trace_rms`
- `log_trace_abs_max`
- `trace_channel_range_mean`
- `trace_outlier_fraction`
- `age_per_token`

## Held-out Eval

- target accuracy: 0.869
- safe prediction rate: 0.906
- mean safe bytes regret: 116.5
- mean predicted total bytes: 3507.3

## Held-out Calibration

- target candidate: M3/affine/4/float16
- feasible subset used: False
- recommended logit offset: -1.50
- target accuracy: 0.725
- safe prediction rate: 0.725
- mean predicted total bytes: 2829.7
