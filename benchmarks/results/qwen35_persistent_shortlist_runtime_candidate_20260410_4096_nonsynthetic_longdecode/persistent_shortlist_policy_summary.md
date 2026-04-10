# Persistent Shortlist Policy

- group by: layer_id, kv_head_id, step_bucket
- policy groups: 36
- compare inputs: 6
- target abs threshold: 0.0500

## Replay Evaluation

- top-1 accuracy: 0.764
- chosen-safe rate: 0.750
- avg selected tokens: 2339.8
- oracle avg selected tokens: 2339.8
- fallback rate: 0.000
- missing bucket rate: 0.000
