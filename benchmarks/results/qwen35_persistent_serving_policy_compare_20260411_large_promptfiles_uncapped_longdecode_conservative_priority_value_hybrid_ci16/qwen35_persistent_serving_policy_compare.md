## Qwen3.5 Conservative Certified Streaming: Uncapped 10-Prompt Longdecode

- Config:
  - `enable_early_exit=true`
  - `full_attention_streaming_order_mode=priority_value_hybrid`
  - `full_attention_streaming_priority_value_upper_weight=0.25`
  - `full_attention_check_interval=16`
  - `full_attention_key_centroid_count_by_layer={19:8,23:16}`
- Result:
  - bias: `2457.43 ms/step`
  - hand-tuned: `3456.50 ms/step`
  - bias faster on `10/10`
- Comparison:
  - prior conservative uncapped bias baseline: `2612.58 ms/step`
  - delta: about `-5.94%`
- Notes:
  - certified early stop triggered on `9/10` prompts
  - `model_roadmap` still ran to full coverage
