# Qwen3.5 Stage 9 `performance_journal` CUDA Tie-Step Diagnostic (2026-04-13)

This note isolates the remaining fixed-tree public-validation residual on CUDA: the late `15`/`16` flip on `performance_journal`.

Artifacts:
- `benchmarks/inspect_qwen35_performance_journal_tie_cuda.py`
- `benchmarks/results/qwen35_performance_journal_tie_cuda_diag_20260413/diagnostic_default.json`
- `benchmarks/results/qwen35_performance_journal_tie_cuda_diag_20260413/diagnostic_default.md`
- `benchmarks/results/qwen35_performance_journal_tie_cuda_diag_20260413/diagnostic_capture.json`
- `benchmarks/results/qwen35_performance_journal_tie_cuda_diag_20260413/diagnostic_capture.md`

## Same-tree tied step

Prompt:
- `docs/performance_journal.md`
- prompt length `2048`
- decode steps `8`
- target tied step `5` (predicts the seventh generated token)

Generated IDs:
- dense: `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- real mixed: `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- non-M0: `[198, 220, 471, 1510, 77518, 28, 15, 7561]`

Pre-argmax readout at the tied step:
- dense:
  - token `15` logit `20.625`, prob `0.3637197018`
  - token `16` logit `20.625`, prob `0.3637197018`
- real mixed:
  - token `15` logit `20.625`, prob `0.3615868688`
  - token `16` logit `20.640625`, prob `0.3672810495`
- non-M0:
  - token `15` logit `20.625`, prob `0.3638806641`
  - token `16` logit `20.625`, prob `0.3638806641`

Immediate read:
- dense and non-M0 remain exactly tied at stored logit precision and resolve to token `15`
- real mixed nudges token `16` above `15` by `0.015625`

## First drift vs non-M0

Real mixed vs non-M0 on the same tree/runtime:
- first nonzero output delta appears at full-attention layer `3`
- target-step logit max-abs delta: `0.0419921875`
- token `15` logit delta: `0.0`
- token `16` logit delta: `0.015625`

Per-layer output max-abs drift after the first split:
- layer `3`: `0.0003662109375`
- layer `7`: `0.00057220458984375`
- layer `11`: `0.00067138671875`
- layer `15`: `0.0008544921875`
- layer `19`: `0.001953125`
- layer `23`: `0.00146484375`

This means the lane-to-lane difference starts before final argmax and then accumulates mildly through later full-attention layers.

## Forced-Capture `final_mix`

To attribute the tied step more precisely, the same CUDA repro was rerun with:
- `DOTCACHE_DISABLE_CUDA_STREAMING_FRONTIER_FAST_PATH=1`
- `--force-stream-attn-capture`

That run captured the per-head logits and gathered values feeding direct-M0/final-mix at the tied step.

Result:
- max captured `final_mix` context-vs-float32-reference delta: `0.00000572`

This is much smaller than the real mixed vs non-M0 lane drift above. The captured direct-M0/final-mix computation therefore matches a float32 recompute on the same inputs very closely.

## Interpretation

The remaining `performance_journal` residual is best explained by tiny upstream mixed-path numeric drift before argmax, likely from accumulation order or similar backend-sensitive ordering effects, not by a `final_mix` helper bug.

This is consistent with the earlier fixed-tree CUDA confidence note and the post-fix MPS reruns:
- the broader round-2 divergence family is gone
- no Stage 9 mixed-only correctness blocker remains
- the leftover public residual is a late tie-boundary sensitivity

Practical next step:
- treat correctness as effectively closed here
- shift CUDA effort back to performance on `final_mix` and `direct_m0_score`, which remain the large cost buckets
