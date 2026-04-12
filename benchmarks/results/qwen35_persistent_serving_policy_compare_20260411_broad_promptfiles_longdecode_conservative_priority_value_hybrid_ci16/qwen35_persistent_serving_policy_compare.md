## Qwen3.5 Conservative Certified Streaming: Broad Promptfiles Longdecode

- Config:
  - `enable_early_exit=true`
  - `full_attention_streaming_order_mode=priority_value_hybrid`
  - `full_attention_streaming_priority_value_upper_weight=0.25`
  - `full_attention_check_interval=16`
  - `full_attention_key_centroid_count_by_layer={19:8,23:16}`
- Result:
  - bias: `3437.43 ms/step`
  - hand-tuned: `4322.04 ms/step`
  - bias faster on `6/6`
- Comparison:
  - prior broad conservative bias baseline: `4480.62 ms/step`
  - delta: about `-23.28%`
- Notes:
  - this is slower than the earlier hybrid `ci4` broad-only run (`3266.26 ms/step`)
  - but `ci16` is the better cross-corpus setting because it also wins on the uncapped long-context set
