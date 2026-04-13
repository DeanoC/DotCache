# Qwen3.5 `performance_journal` Residual After Handoff Fix (CUDA, 2026-04-13)

This note records the remaining `performance_journal` investigation after the
logical-sequence-length handoff fix.

## Case

- prompt: `docs/performance_journal.md`
- prompt length: `2048`

## Main result

On CUDA after the handoff fix, dense and serving now match exactly on the full
8-token continuation for `performance_journal`:

- dense:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- serving:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`

So there is no remaining CUDA-side serving divergence on this case.

## Residual localization

The only remaining public discrepancy reported from the refreshed MPS reruns is
the late tail:

- MPS dense:
  - `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- MPS serving:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`

On CUDA, the disputed step is an argmax-tie boundary.

At generated token step `6`:

- dense top candidates:
  - `16`, `15`, `17`, `19`, `23`
- dense logits:
  - `logit[15] = 20.625`
  - `logit[16] = 20.625`
- serving logits:
  - `logit[15] = 20.625`
  - `logit[16] = 20.625`

So on CUDA the `15` vs `16` choice is exactly tied at the stored logit
precision, and `argmax` resolves to token `15`.

## Dense vs serving logit drift

Dense and serving are also very close numerically across the full sequence:

- maximum dense-vs-serving logit delta by step stayed in roughly the
  `0.0156` to `0.0205` range
- at the disputed step, the dense-vs-serving deltas for the two candidate
  tokens were exactly:
  - `delta(logit[15]) = 0.0`
  - `delta(logit[16]) = 0.0`

That means the remaining public `performance_journal` difference is not caused
by the serving path selecting a different local winner on CUDA.

## Interpretation

Best current read:

- the substantive public correctness problem was the handoff logical-sequence-length bug
- after fixing that, `performance_journal` reduces to a late backend-sensitive
  dense-alignment issue
- the residual `15` vs `16` tail flip is consistent with tiny backend numeric
  differences around a near-tied logit pair, not with a remaining Stage 9 mixed
  execution bug
