# Qwen3.5 Exact-Key Live Policy Compare

This compares simple layer-15 policies in the live real-mixed Stage 9 runtime.

## Ranked policies

- `baseline`:
  - description: Current global policy with no layer-15 override.
  - overall avg ms/step: 683.9923
  - exact-match vs baseline: 1.000
  - per-manifest avg ms/step: {"/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_broad_20260412.json": 838.6311699832731, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_external_20260412.json": 464.9992725462653, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_large_20260412.json": 694.3592406256357, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json": 621.5715703146998}
- `layer15_always_024`:
  - description: Always set layer 15 to 0.24.
  - overall avg ms/step: 694.6162
  - exact-match vs baseline: 1.000
  - per-manifest avg ms/step: {"/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_broad_20260412.json": 847.8005712264954, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_external_20260412.json": 468.7403384668869, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_large_20260412.json": 695.6256661032967, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json": 652.6871458978954}
- `layer15_len_ge_1800_024`:
  - description: Use layer 15 -> 0.24 only when prompt length is at least 1800 tokens.
  - overall avg ms/step: 723.1005
  - exact-match vs baseline: 1.000
  - per-manifest avg ms/step: {"/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_broad_20260412.json": 875.4969992041879, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_external_20260412.json": 516.9450919590114, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_large_20260412.json": 741.4143062225776, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json": 643.2587212754394}
- `layer15_code_or_len_ge_1800_024`:
  - description: Use layer 15 -> 0.24 for code files or prompts of at least 1800 tokens.
  - overall avg ms/step: 756.6313
  - exact-match vs baseline: 1.000
  - per-manifest avg ms/step: {"/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_broad_20260412.json": 859.0582967917726, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_external_20260412.json": 481.9326198315442, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_large_20260412.json": 768.8377182341355, "/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json": 771.209749190651}
