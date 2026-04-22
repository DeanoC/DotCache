# Paper-2 skip-path ablation (archived)

These are the first-pass certified results produced BEFORE the
hybrid-kernel dispatch fix (commit `ac87e753`). They ran the
**block-skipping** algorithm (Paper 2 semantics): tail blocks outside
the adaptive top-K* set were masked to `-inf` in SDPA, dropping their
mass entirely from the attention output.

That is NOT the algorithm the paper describes. Paper 1 attends every
block, using INT8 keys on the tail and FP16 keys on the top-K* set.
The corrected certified results live one directory up.

Kept here as an ablation comparison: "what happens if we hard-drop
tail blocks instead of attending them with INT8 keys?". The table
below summarises the delta between the two algorithms on the same
dense baselines:

| Cell | Δacc vs dense (skip) | Paper-1 hybrid |
|---|---|---|
| 05 niah 4K | +3.33pp (0 crit)  | (see main dir) |
| 11 niah 8K | −6.67pp (3 crit)  | (see main dir) |
| 17 niah 16K | −13.33pp (5 crit) | (see main dir) |
| 04 pg19 4K  | Δppl +0.009 | (see main dir) |
| 10 pg19 8K  | Δppl +0.005 | (see main dir) |
| 16 pg19 16K | Δppl +0.021 | (see main dir) |
| 06 ruler 4K  | Δacc −0.003 | (see main dir) |
| 12 ruler 8K  | Δacc −0.003 | (see main dir) |
| 18 ruler 16K | Δacc −0.003 | (see main dir) |

The "skip path beating dense on NIAH 4K" was an artefact: dropping
low-mass tail blocks concentrates softmax mass on the needle, which
is anti-dense but helpful for retrieval. With every block attended
(Paper 1), the tail's INT8-noisy scores redistribute mass away from
the needle — the §9 non-monotonicity the paper predicts.
