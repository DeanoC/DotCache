## Qwen3.5 Conservative Certified Streaming: External Promptfiles Longdecode

- Config:
  - `enable_early_exit=true`
  - `full_attention_streaming_order_mode=priority_value_hybrid`
  - `full_attention_streaming_priority_value_upper_weight=0.25`
  - `full_attention_check_interval=16`
  - `full_attention_key_centroid_count_by_layer={19:8,23:16}`
- Result:
  - bias: `2412.91 ms/step`
  - hand-tuned: `3879.88 ms/step`
  - bias faster on `3/3`
- Comparison:
  - prior external conservative bias baseline: `4867.92 ms/step`
  - delta: about `-50.43%`
- Notes:
  - certified early stop triggered on all `3/3` prompts
