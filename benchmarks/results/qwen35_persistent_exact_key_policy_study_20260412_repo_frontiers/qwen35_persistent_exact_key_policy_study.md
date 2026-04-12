# Qwen3.5 Exact-Key Policy Study

This compares simple layer-15 policy choices against the checked-in exact-key frontier studies.

## Ranked policies

- `layer15_always_020`:
  - description: Always set layer 15 to 0.20.
  - overall avg ms/step: 1489.8605
  - per-corpus avg ms/step: {"qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad": 1772.4438445632888, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external": 840.1664982544995, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large": 1515.2186235274712}
  - chosen counts: {"0.20": 19}
- `layer15_len_le_2048_else_024`:
  - description: Use 0.20 when prompt length is <= 2048, otherwise 0.24.
  - overall avg ms/step: 1504.1021
  - per-corpus avg ms/step: {"qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad": 1723.6683705972002, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external": 840.1664982544995, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large": 1571.5429969386605}
  - chosen counts: {"0.20": 15, "0.24": 4}
- `layer15_len_le_1536_else_024`:
  - description: Use 0.20 when prompt length is <= 1536, otherwise 0.24.
  - overall avg ms/step: 1524.2111
  - per-corpus avg ms/step: {"qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad": 1684.634925390128, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external": 840.1664982544995, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large": 1633.170096885442}
  - chosen counts: {"0.20": 8, "0.24": 11}
- `layer15_always_024`:
  - description: Always set layer 15 to 0.24.
  - overall avg ms/step: 1543.0048
  - per-corpus avg ms/step: {"qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad": 1693.054104204445, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external": 847.6201996624392, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large": 1661.5905838589242}
  - chosen counts: {"0.24": 19}
- `baseline`:
  - description: Current global policy with no layer-15 override.
  - overall avg ms/step: 1593.6701
  - per-corpus avg ms/step: {"qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad": 1864.5783203004005, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external": 841.938093707237, "qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large": 1656.6447323028115}
  - chosen counts: {"baseline": 19}
