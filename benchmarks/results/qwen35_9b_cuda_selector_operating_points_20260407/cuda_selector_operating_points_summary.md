# Qwen3.5-9B CUDA Selector Operating Points

This is a partial operating-points summary. Any point or pack marked `partial` or `missing` did not finish and is reported without final metrics.

## Recommendation

- Matched-quality point: `op_weighted_dense_control`
- Smallest-memory acceptable point: `op_weighted_dense_control`
- Fastest acceptable point: `op_weighted_dense_control`
- Recommended CUDA default: `op_weighted_dense_control`

| selection | point | floor | dense_match_rate | accuracy_when_dense_correct | official_score | decode_ms | eff_bytes/token |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched_quality | op_weighted_dense_control | weighted | 0.250 | 0.556 | 0.332 | 117.9 | 49116.2 |
| smallest_memory_acceptable | op_weighted_dense_control | weighted | 0.250 | 0.556 | 0.332 | 117.9 | 49116.2 |
| fastest_acceptable | op_weighted_dense_control | weighted | 0.250 | 0.556 | 0.332 | 117.9 | 49116.2 |
| recommended_default | op_weighted_dense_control | weighted | 0.250 | 0.556 | 0.332 | 117.9 | 49116.2 |

## Completion Status

| point | pack | status | detail |
| --- | --- | --- | --- |
| op_floor_070 | task_compare | complete | task compare artifacts present |
| op_floor_070 | longbench_mini | complete | merged longbench outputs present |
| op_floor_070 | longbench_lb21_16_smoke_20260406 | missing | longbench outputs missing |
| op_floor_075 | task_compare | complete | task compare artifacts present |
| op_floor_075 | longbench_mini | complete | merged longbench outputs present |
| op_floor_075 | longbench_lb21_16_smoke_20260406 | missing | longbench outputs missing |
| op_floor_cluster_095_090_085_080 | task_compare | complete | task compare artifacts present |
| op_floor_cluster_095_090_085_080 | longbench_mini | complete | merged longbench outputs present |
| op_floor_cluster_095_090_085_080 | longbench_lb21_16_smoke_20260406 | partial | shard_00.jsonl=502, shard_01.jsonl=512 |
| op_weighted_dense_control | task_compare | complete | task compare artifacts present |
| op_weighted_dense_control | longbench_mini | complete | merged longbench outputs present |
| op_weighted_dense_control | longbench_lb21_16_smoke_20260406 | complete | merged longbench outputs present |

## Distinct Operating Points

| point | strategy_id | floor | offline_safe | offline_target | offline_bytes |
| --- | --- | --- | --- | --- | --- |
| op_floor_070 | linear_softmax_compression_floor_0p70_dense_control | 0.70 | 0.668 | 0.655 | 4329.9 |
| op_floor_075 | linear_softmax_compression_floor_0p75_dense_control | 0.75 | 0.770 | 0.724 | 4673.7 |
| op_floor_cluster_095_090_085_080 | linear_softmax_compression_floor_0p95_dense_control | 0.95 | 0.627 | 0.618 | 3752.7 |
| op_weighted_dense_control | linear_softmax_compression_weighted_dense_control | weighted | 1.000 | 1.000 | 4854.5 |

## task_compare Systems

| point | floor | status | dense_match_rate | accuracy_when_dense_correct | score | error_vs_exact | decode_ms | eff_bytes/token | resident_mib | v_m0 | v_m3 | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op_floor_070 | 0.70 | complete | 1.000 | 1.000 | 0.667 | 0.000 | 181.5 | 46449.2 | 67.4 | 267.8 | 2804.2 | fit |
| op_floor_075 | 0.75 | complete | 1.000 | 1.000 | 0.667 | 0.000 | 354.8 | 18752.0 | 27.4 | 3072.0 | 0.0 | fit |
| op_floor_cluster_095_090_085_080 | 0.95 | complete | 1.000 | 1.000 | 0.667 | 0.000 | 351.8 | 18752.0 | 27.4 | 3072.0 | 0.0 | fit |
| op_weighted_dense_control | weighted | complete | 1.000 | 1.000 | 0.667 | 0.000 | 355.1 | 18752.0 | 27.4 | 3072.0 | 0.0 | fit |

## task_compare Quality

| point | floor | status | dense_match_rate | accuracy_when_dense_correct | score | error_vs_exact | decode_ms | eff_bytes/token | resident_mib | v_m0 | v_m3 | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op_floor_070 | 0.70 | complete | 1.000 | 1.000 | 0.667 | 0.000 | 581.5 | 36901.8 | 53.9 | 963.2 | 2108.8 | fit |
| op_floor_075 | 0.75 | complete | 1.000 | 1.000 | 0.667 | 0.000 | 355.6 | 18752.0 | 27.4 | 3072.0 | 0.0 | fit |
| op_floor_cluster_095_090_085_080 | 0.95 | complete | 1.000 | 1.000 | 0.667 | 0.000 | 352.3 | 18752.0 | 27.4 | 3072.0 | 0.0 | fit |
| op_weighted_dense_control | weighted | complete | 1.000 | 1.000 | 0.667 | 0.000 | 355.0 | 18752.0 | 27.4 | 3072.0 | 0.0 | fit |

## longbench_mini Systems

| point | floor | status | dense_match_rate | accuracy_when_dense_correct | score | error_vs_exact | decode_ms | eff_bytes/token | resident_mib | v_m0 | v_m3 | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op_floor_070 | 0.70 | complete | 0.625 | 1.000 | 0.465 | 0.250 | 2634.5 | 31406.9 | 155.9 | 1647.1 | 9140.9 | fit |
| op_floor_075 | 0.75 | complete | 0.625 | 1.000 | 0.465 | 0.250 | 1976.9 | 32682.5 | 162.9 | 1114.1 | 9673.9 | fit |
| op_floor_cluster_095_090_085_080 | 0.95 | complete | 0.625 | 1.000 | 0.465 | 0.250 | 4506.6 | 30041.5 | 147.7 | 3426.6 | 7365.4 | fit |
| op_weighted_dense_control | weighted | complete | 0.625 | 1.000 | 0.465 | 0.250 | 120.3 | 46149.7 | 228.6 | 0.0 | 10788.0 | fit |

## longbench_mini Quality

| point | floor | status | dense_match_rate | accuracy_when_dense_correct | score | error_vs_exact | decode_ms | eff_bytes/token | resident_mib | v_m0 | v_m3 | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op_floor_070 | 0.70 | complete | 0.625 | 1.000 | 0.465 | 0.250 | 3869.0 | 29504.3 | 145.1 | 5711.5 | 5080.5 | fit |
| op_floor_075 | 0.75 | complete | 0.500 | 1.000 | 0.481 | 0.375 | 4705.6 | 30190.7 | 149.1 | 4589.2 | 6202.8 | fit |
| op_floor_cluster_095_090_085_080 | 0.95 | complete | 0.875 | 1.000 | 0.465 | 0.000 | 3391.3 | 24649.1 | 123.3 | 7576.5 | 3215.5 | fit |
| op_weighted_dense_control | weighted | complete | 0.625 | 1.000 | 0.465 | 0.250 | 120.9 | 46149.7 | 228.6 | 0.0 | 10788.0 | fit |

## longbench_lb21_16_smoke_20260406 Systems

| point | floor | status | dense_match_rate | accuracy_when_dense_correct | score | error_vs_exact | decode_ms | eff_bytes/token | resident_mib | v_m0 | v_m3 | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op_floor_070 | 0.70 | missing | - | - | - | - | - | - | - | - | - | - |
| op_floor_075 | 0.75 | missing | - | - | - | - | - | - | - | - | - | - |
| op_floor_cluster_095_090_085_080 | 0.95 | partial | - | - | - | - | - | - | - | - | - | - |
| op_weighted_dense_control | weighted | complete | 0.250 | 0.556 | 0.332 | 0.500 | 117.9 | 49116.2 | 202.7 | 0.0 | 9515.4 | fit |

## longbench_lb21_16_smoke_20260406 Quality

| point | floor | status | dense_match_rate | accuracy_when_dense_correct | score | error_vs_exact | decode_ms | eff_bytes/token | resident_mib | v_m0 | v_m3 | fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| op_floor_070 | 0.70 | missing | - | - | - | - | - | - | - | - | - | - |
| op_floor_075 | 0.75 | missing | - | - | - | - | - | - | - | - | - | - |
| op_floor_cluster_095_090_085_080 | 0.95 | partial | - | - | - | - | - | - | - | - | - | - |
| op_weighted_dense_control | weighted | complete | 0.250 | 0.556 | 0.332 | 0.500 | 116.5 | 49116.2 | 202.7 | 0.0 | 9515.4 | fit |

## Paper Note

Dense preservation remains the control objective. The recommended CUDA default is `op_weighted_dense_control` because it stays inside the acceptable dense-preservation envelope on the stronger LongBench smoke pack while minimizing serving latency.
If the paper wants the cleanest matched-quality row, use `op_weighted_dense_control`. If the paper wants the clearest memory story, use `op_weighted_dense_control` alongside the recommended default.
Primary recommendation metrics on the stronger smoke pack: dense_match_rate=0.250, accuracy_when_dense_correct=0.556, official_score=0.332, decode_ms=117.9, effective_bytes_per_token=49116.2.
