# Runtime Page Selector Artifact

- classes: 2
- training examples: 352
- runtime feature count: 32
- artifact path: benchmarks/results/qwen35_selector_qwen35_0p8b_local_20260406/serving_selector_artifact_family_reasoning_holdout_v3/linear_selector_model.json
- class balance exponent: 0.50
- safe bytes weight: 1.00
- reference candidate: M3/affine/4/float16

## Features

- `stage_decode`
- `kind_key`
- `query_present`
- `layer_fraction`
- `kv_head_fraction`
- `log_sequence_length`
- `log_token_start`
- `log_token_age`
- `token_count`
- `head_dim`
- `trace_rms`
- `log_trace_abs_max`
- `trace_channel_range_mean`
- `trace_outlier_fraction`
- `age_per_token`
- `token_end_fraction`
- `token_age_fraction`
- `age_bucket_ge_64`
- `age_bucket_ge_256`
- `age_bucket_ge_1024`
- `sequence_length_ge_512`
- `sequence_length_ge_1024`
- `sequence_length_ge_2048`
- `decode_old_page_indicator`
- `decode_long_context_indicator`
- `decode_key_indicator`
- `family_instruction`
- `family_retrieval`
- `variant_constraints`
- `variant_formatting`
- `variant_memo`
- `variant_transcript`

## Held-out Eval

- target accuracy: 0.881
- safe prediction rate: 0.912
- mean safe bytes regret: 106.1
- mean predicted total bytes: 3507.3

## Held-out Calibration

- target candidate: M3/affine/4/float16
- feasible subset used: False
- recommended logit offset: -1.50
- target accuracy: 0.719
- safe prediction rate: 0.719
- mean predicted total bytes: 2873.7
