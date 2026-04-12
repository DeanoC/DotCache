## Qwen3.5 Stage 9 Packed Mixed Certified Streaming: External Promptfiles Longdecode

- Config:
  - `enable_early_exit=true`
  - `enable_full_attention_mixed_mode_execution=true`
  - `full_attention_mixed_mode_execution_strategy=direct_m0_metal_packed`
  - `full_attention_mixed_mode_execution_max_k_comp_error=0.10`
  - `full_attention_streaming_order_mode=priority_value_hybrid`
  - `full_attention_streaming_priority_value_upper_weight=0.25`
  - `full_attention_check_interval=16`
  - `full_attention_key_centroid_count_by_layer={19:8,23:16}`
- Result:
  - bias: `1232.69 ms/step`
  - hand-tuned: `1563.89 ms/step`
  - bias faster on `3/3`
- Comparison:
  - conservative certified external bias: `2412.91 ms/step`
  - delta: about `-48.91%`
- Notes:
  - certified early stop triggered on all `3/3` prompts
