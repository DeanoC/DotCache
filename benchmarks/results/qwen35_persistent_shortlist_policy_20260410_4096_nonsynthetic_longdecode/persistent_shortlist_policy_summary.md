# Persistent Shortlist Policy

- group by: layer_id, kv_head_id, prompt_family, step_bucket
- policy groups: 108
- compare inputs: 6
- target abs threshold: 0.0500

## Replay Evaluation

- top-1 accuracy: 0.861
- chosen-safe rate: 0.757
- avg selected tokens: 2339.8
- oracle avg selected tokens: 2339.8
- fallback rate: 0.000
- missing bucket rate: 0.000
