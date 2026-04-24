# Paper ↔ Code Audit — `Certified_Quantised_Attention.tex`

**Date:** 2026-04-24 (**revised after user flagged sloppiness in first pass**)
**Paper commit:** `43025cd8` on `bench/overnight-20260423`
**Code snapshot:** this branch, HEAD

## 0. Correction of errors in the first pass

My first pass of this audit (committed in `396658ab`) claimed several artifacts were "missing" or "unsupported" without checking the repository. The user was right to flag this. Specifically, I made these wrong claims:

1. **"R-5 NIAH 8K 100-paired-trial follow-up: artifact not found"** — **WRONG.** The artifact is at `benchmarks/results/niah_8k_tau_sweep_20260422/`, committed `cfe95d91` / `2b038240` / `debd1d5e` on 2026-04-22. Three τ_cov values × 100 paired trials each, plus a detailed `SUMMARY.md`.
2. **"R-2 32K dense ppl 10.340: unresolved provenance"** — **PARTLY WRONG.** `10.340` matches the running dense ppl at chunk 10 of the 32K rerun log (`pg19_ctx32768.log:15: Dense [10/20]: ppl=10.34`). Either the paper author read the intermediate instead of the final (`Dense [20/20]: ppl=9.57`), or there is a separate 10-chunk 32K run that I have not located. It is not unsupported; it just has a discrepancy I need to identify more precisely.
3. **RULER paper data** — I did not check `benchmarks/results/ruler_paper_20260417/`, which has dedicated 4K/8K RULER runs with 50 samples × 7 subtasks, directly feeding the paper's Tab. 4 numbers.
4. **LongBench paper data** — I did not check `benchmarks/results/longbench_paper_20260418/`, which has 4K/8K calibrated + eps0 + chunked_prefill artifacts.

These were **process failures** — I grepped on my current context rather than `ls`-ing the results tree and reading the directories that landed on 2026-04-16 → 2026-04-22. The revised audit below walks every paper claim against an actual file on HEAD.

Legend: ✅ match · ❌ mismatch · ⚠️ partial / paper-imprecise · ℹ️ note

## 1. Key quantisation (§2.3, §5.2)

| # | Paper claim (line) | Code evidence | Verdict |
|---|---|---|---|
| K-1 | §2.3 L218-222: per-channel INT8 with scale $s_c$ **and zero point $z_c$**, range **[−128, 127]** | `dotcache/kernels/tiered_kv_cache.py:122-133` — **symmetric**, scale-only, range **[−127, 127]**. Comment L122: *"Per-channel symmetric INT8 quantisation"*. No zero-point buffer anywhere in `tiered_kv_cache.py`. | ❌ |
| K-2 | §2.3 L224: quantisation after RoPE | `tiered_kv_cache.py:100` reads `past_key_values` (HF stores post-RoPE keys for Llama) | ✅ |
| K-3 | §2.3 L228: per-block per-channel scales **and zeros**, 2d FP32 per block, immutable | `tiered_kv_cache.py:168-171` allocates `scale_buf = [kv_heads, max_blocks, head_dim] float32` only. No zero-point buffer. Per-block immutable ✓. | ❌ (no zeros) |
| K-4 | §2.3 L228: trailing partial block stays FP16 until $B$ tokens arrive | `certified_attention.py:682-690` handles trailing partial block via hybrid FP16 path | ✅ |
| K-5 | §5.2 L572-578 scoring: $q\cdot\hat{k} = \sum_c(q_c s_c)k^{int8}_{t,c} + \sum_c q_c z_c^{(b)}$ (depends on zeros) | `selective_attend_triton.py:160-164` Triton kernel: `k_tile = k_int8_tile.to(fp32) * ch_scale` (scale-only). No zero-point term. | ❌ |
| K-6 | §4.2 L445: tight per-block $\Delta_b = (1/(2\sqrt{d}))\sum_c \|q_c\| s_c^{(b)}$; implementation uses conservative $\max_b \Delta_b$ | `certified_attention.py:466-469` computes `per_channel_scale = key_scales.amax(dim=1)` then `Σ\|q_c\| × s_c^max / (2√d)`. This is ≥ paper's `max_b Δ_b` (looser but valid). | ⚠️ |

**Implication.** The paper's key quantisation is **asymmetric** per-channel; the code is **symmetric** per-channel. The scoring-formula constant term in §5.2 doesn't exist in the kernel. Theorem 2 still holds for the symmetric scheme but the derivation needs rewording. Every paper run (arxiv_v1, rerun, 100-trial sweep) used the symmetric code, so NO data reflects the paper's asymmetric scheme.

## 2. Value quantisation (§2.3, §3.1, §4.1, §7)

| # | Paper claim (line) | Code evidence | Verdict |
|---|---|---|---|
| V-1 | §2.3 L230-234: per-group INT4 values, $g{=}16$, FP16 scale+zero, 2 INT4/byte | `tiered_kv_cache.py:221-250` `from_fp16_cache_int4v` + `selective_attend_triton.py:245-370` INT4 kernel. All present. Default `group_size=16`. | ✅ (exists) |
| V-2 | §3.1, §7 L667: Certified config uses **INT4 values $g{=}16$** | **NO paper bench has used INT4 values.** All four benches (`pg19_perplexity.py:262`, `niah.py:167`, `ruler.py:401`, `longbench.py:305`) call `create_tiered_cache_from_model` (FP16 values). Verified no `v_format`, `int4`, `eta_int4`, `e_val` occurrences in any committed JSON under `arxiv_v1_20260420/`, `ruler_paper_20260417/`, `longbench_paper_20260418/`, `niah_8k_tau_sweep_20260422/`, `pg19_rerun_20260423/`, or `pg19_ekey_short_20260423/`. | ❌ **CRITICAL** |
| V-3 | §2.3 L236: per-block $\eta_b$ and $\nu_b = \max_{t\in b}\|V_t\|_2$ | `tiered_kv_cache.py:246-249` writes `values_int4_errors` **only on int4v path** (unused in paper runs). `ν_b` (block value norm) is **not computed anywhere** in the repo. | ⚠️ |
| V-4 | §4.1 Thm 1: $E_{val}\leq\eta$; Cor: $E_{val}\leq\sum_b\rho_b\eta_b$ | `certified_attention.py:124-173` `compute_value_error_bound` implements the blockwise form, called only in the int4v branch (L1118). **Zero calls on the paper-bench runs.** | ⚠️ (unexercised) |
| V-5 | §3.4 L338 Rung-2: promote blocks where $\hat\rho_b\eta_b > v_{tol}$ to FP16 values | `certified_attention.py:1118-1171` Rung-2. Gated on `cache.values_int4_packed is not None`. **Cannot fire** on paper-bench runs. | ⚠️ |

## 3. Adaptive $K^*$, fallback ladder, bounds (§3.3, §3.4, §4, §6)

| # | Paper claim (line) | Code evidence | Verdict |
|---|---|---|---|
| A-1 | τ_cov default 0.995 | `certified_attention.py:37`; overnight/arxiv_v1 pass `--tau-cov 0.995` | ✅ |
| A-2 | K_min=2, K_max=128 (§7 L667) | `certified_attention.py:38-40`; benches pass explicitly | ✅ |
| A-3 | §3.3 L327: true tail ≤ $e^{2\Delta}(1-\tau_{cov})$ | Implemented in `compute_tier2_residual_mass` / mass bound | ✅ |
| A-4 | §4.2 L464 & §4.4 L517: implementation substitutes $e^{3\Delta}$ for optional INT8 query scoring | Runtime code uses FP16 query (Triton loads Q as FP32 at `selective_attend_triton.py:157`). The $e^{3\Delta}$ substitution is thus conservative, not strictly necessary for the measured path. Not telemetrised. | ⚠️ |
| A-5 | Rung-1: expand K\* | `certified_attention.py:743-760`, `rung1_multiplier=2.0` | ✅ |
| A-6 | Rung-2: INT4→FP16 value | Exists but unreachable on FP16-values runs | ❌ in practice |
| A-7 | Rung-3: per-head full FP16 via torch SDPA | `certified_attention.py:1178-1194` with `recompute_heads_dense_fp16` | ✅ |
| A-8 | Rung-4: full FP16 all-heads SDPA when score-consistency violates | `certified_attention.py:876-929`. Gated on `score_consistency_check=True`. arxiv_v1 + `ruler_paper_20260417` + `longbench_paper_20260418` + `niah_8k_tau_sweep_20260422` **all pass `--score-consistency-check`**; overnight rerun + in-flight 128K do **not**. | ✅ (arxiv_v1) / ⚠️ (rerun/128K unarmed) |
| A-9 | Exploration 1-5% (§6 L591) | `exploration_rate=0.02` passed by every paper runner | ✅ |
| A-10 | §6.1 L614-621 boundary verification Eq. 30 | Not implemented — code only compares rankings *within* the promoted set. Grep for `ell_b_int8 + delta` or equivalent: no match. | ❌ |
| A-11 | §6 L589 score consistency per-**token** | `score_consistency_violations` (`certified_attention.py:476-493`) operates per-**block** over top-K blocks, not per-token over all tokens. | ⚠️ (different granularity, but §6.1 boundary discussion is per-block, so paper is internally mixed) |
| A-12 | §4.5 L528 FP32 accumulators; §3.2 L309 FP64 online-softmax scalars | Kernel online-softmax state FP64 (`selective_attend_triton.py:141-143`); output accumulator FP32 | ✅ (stricter than stated) |

## 4. §7 Certified config vs runners — what was actually measured

| # | Paper §7 claim | Runner reality | Verdict |
|---|---|---|---|
| C-1 | L659 Model: **LLaMA 3.1-8B-Instruct** | All paper benches default to `NousResearch/Meta-Llama-3.1-8B` (non-Instruct base). `run_arxiv_v1_sweep.py:272-277` explicit comment: *"The spec calls for meta-llama/Llama-3.1-8B-Instruct; the benchmark scripts default to NousResearch's non-gated mirror"*. `ruler_paper_20260417/calibrated_4k.log` and `niah_8k_tau_sweep_20260422/niah_8k_tau0995_n100.log` both confirm the non-Instruct base was used. | ❌ **ALL runs** |
| C-2 | L667 INT8 keys + **INT4 values $g{=}16$** | INT8 keys **symmetric** + **FP16 values** on every committed paper-related JSON | ❌ (V-2) + ❌ (K-1) |
| C-3 | τ_cov=0.995, K_min=2, K_max=128, v_tol=0.05, r=1 | Flags passed exactly by arxiv_v1, RULER paper, NIAH 100-trial, LongBench paper, and overnight rerun. `v_tol=0.05` is unused (FP16-values runs). | ✅ selector side / ⚠️ v_tol |
| C-4 | FP64 online-softmax accumulators | Verified in Triton kernels | ✅ |
| C-5 | L669 PG-19: **5 books, non-overlapping windows** | arxiv_v1 used `--num-chunks=5` for 4K/8K/16K dense (cells 01/07/13) and 4K/8K/16K cert (cells 04/10/16); 32K used `--num-chunks=20` (cells 19/22) plus the overnight rerun's 20-chunk replication. Paper §8.2 Tab. 2 caption correctly says "4K-16K: 5 books; 32K: 20 chunks". | ✅ |
| C-6 | L669 NIAH: **5 needles, 30 trials per context, plus 100-paired-trial follow-up at 8K** | **arxiv_v1 cells 05/11/17/23 used `--needles 3`** (verified in `arxiv_v1_20260420/05_niah_4K_certified.log:1`: *"NIAH: contexts=[4]K, needles=3"*). The 100-trial sweep (`niah_8k_tau_sweep_20260422/`) used `--needles 10` (5 orig + 5 harder). **No run used `--needles 5`.** The paper's "5 needles" phrasing most plausibly refers to the "orig 5" subset of the 100-trial sweep (`NEEDLES[0..4]`); the "30 trials per context" applies to arxiv_v1's 3-needle runs. | ⚠️ (paper phrasing conflates two subsets, not unsupported) |
| C-7 | L669 RULER: **7 subtasks × 50 samples per context** | `ruler_paper_20260417/summary.csv` confirms 50 samples per subtask per config. arxiv_v1 cells 03/06/09/12/15/18/21/24 also used `--num-samples 50` via the full-sweep config. | ✅ |
| C-8 | L669 Contexts: **4K, 8K, 16K, 32K** | arxiv_v1 covers all four. `ruler_paper_20260417` covers only 4K/8K; 16K/32K from arxiv_v1. 128K in-flight — outside paper scope unless text is updated. | ✅ |

## 5. Results tables — numerical sources

| # | Paper claim (line) | Source file / run | Verdict |
|---|---|---|---|
| R-1 | Tab. 1 (L691) PG-19 Δppl: 4K +0.011 / 8K +0.003 / 16K +0.002 / **32K −0.002** | arxiv_v1 cells 04/10/16/22: +0.0114 / +0.0030 / +0.0017 / **−0.0042**. 4K/8K/16K match after rounding. **32K paper −0.002 is half the arxiv_v1 value −0.0042** — rounding inconsistency OR paper used a different 32K run. See R-2. | ⚠️ |
| R-2 | Tab. 2 (L713) PG-19 **dense ppl 32K = 10.340** | No JSON under `benchmarks/results/` contains `10.340` as a final 32K dense perplexity. The ONLY occurrence of `10.34` is the **chunk-10 intermediate running ppl** in `pg19_rerun_20260423/pg19_ctx32768.log:15`: `Dense [10/20]: ppl=10.34`. The rerun's final at chunk 20 was 9.57. arxiv_v1 cell 19 final was 11.0481. **The paper's 10.340 either reads an intermediate running value OR comes from a 10-chunk 32K run not committed to this repo.** | ❌ (needs resolution) |
| R-3 | Tab. 3 (L729) 20-chunk replication: 4K +0.009 / 8K −0.001 / 16K +0.002 / 32K −0.002 | 4K from `overnight_20260423/pg19_ctx4096.json`: +0.0085 ≈ +0.009 ✓. 8K from `pg19_rerun_20260423/pg19_ctx8192.json`: **−0.0014 ≈ −0.001** ✓. 16K from `pg19_ctx16384.json`: **+0.0017 ≈ +0.002** ✓. 32K from `pg19_ctx32768.json`: **−0.0017 ≈ −0.002** ✓. | ✅ |
| R-4 | Tab. 4 (L755) RULER dense/cert 0.955 / 0.930 / 0.898 / 0.886 → 0.955 / 0.933 / 0.905 / 0.888 | 4K/8K: `ruler_paper_20260417/calibrated_4k.json`, `calibrated_8k.json`; also in arxiv_v1 cells 03/09. 16K/32K: arxiv_v1 cells 15/18/21/24. All numbers match after 3-digit rounding. | ✅ |
| R-5 | Tab. 5 (L776) NIAH 4K dense 83.3% / cert 76.7%; 16K 70.0/73.3; 32K 70.0/70.0 | `arxiv_v1_20260420/05_niah_4K_certified.json` (`dense_accuracy=0.8333`, `certified_accuracy=0.7667`), cells 14/17/20/23 at 16K/32K. All match. | ✅ |
| R-6 | Tab. 5 8K pooled 100 trials: Δ=−2%, 95% CI [−7,+3], **McNemar p=0.38** | `niah_8k_tau_sweep_20260422/SUMMARY.md` at τ=0.995: **pooled 10-needle n=100 gives Δ=−0.020, 95% CI [−0.070, +0.030], McNemar p=0.69**. The **p=0.38** figure matches the **orig-5 subset (n=50): Δ=−0.060, CI [−0.160, +0.020], p=0.38**. Paper appears to pool the 10-needle CI with the 5-needle subset p-value. | ❌ (p-value inconsistency) |
| R-7 | §8.3 "Δ=+8pp on harder needles at τ=0.999" | `niah_8k_tau_sweep_20260422/SUMMARY.md`: τ=0.999 new-5 subset Δ=+0.080, CI [+0.020, +0.160], discordant 4/0, McNemar exact p=0.125. Matches paper. | ✅ |
| R-8 | §8.4 L922 score-consistency canary: 0 violations across all 12 certified cells | Verified across arxiv_v1 cert cells + `ruler_paper_20260417/` + `niah_8k_tau_sweep_20260422/` (all pass `--score-consistency-check`). Overnight rerun + in-flight 128K did NOT pass the flag, but those runs aren't what the paper's "12 cells" refers to. | ✅ |
| R-9 | §8.4 L922: ranking-consistency ~0.12%/head/step on NIAH | arxiv_v1 NIAH cells report `ranking_fallback_rate` in `native.ranking_fallback_summary`; matches. | ✅ |
| R-10 | §8.4 L922: boundary verification 0 triggers | Boundary check Eq. 30 is not implemented (A-10). "0 triggers" is vacuous because the check isn't running. | ❌ |

## 6. What IS supported

- Paper-1 hybrid attend-all kernel (every block contributes; top-K\* FP16 keys, tail INT8 keys) ✅
- Adaptive K\* with τ_cov, K_min, K_max, ranking_r, exploration 2%, Rung-1/3/4 ✅
- FP64 online-softmax scalars, FP32 output accumulator ✅
- RULER paper table numbers (Tab. 4), dense and certified, at all four contexts ✅
- PG-19 Tab. 3 (20-chunk replication) at 4K/8K/16K/32K matches our reruns ✅
- NIAH Tab. 5 underpowered cells (4K, 16K, 32K) match arxiv_v1 ✅
- NIAH 8K harder-5 τ=0.999 finding matches 100-trial sweep ✅
- Score-consistency canary 0 violations on arxiv_v1 + dedicated paper runs ✅

## 7. Discrepancies that actually require action

Ranked by blast radius. The first two are text edits; the remainder require code or re-runs.

### 7.1 Text-only fixes (no code, no re-run)

| # | Fix needed | Paper location |
|---|---|---|
| T-1 | "5 needles × 30 trials" → either "3 needles × 30 trials (10 depths × 3 needles)" for the arxiv_v1 underpowered cells, OR restate as "5 needles for the orig subset within the 100-trial 8K run". | §7 L669 |
| T-2 | Reconcile R-6 p-value: either change p=0.38 to p=0.69 (matching pooled 100-trial) OR change Δ=−2% to Δ=−6% (matching orig-5 subset). Currently the paper pairs pooled CI with subset p. | §8.1 L682, §8.3 L782 |
| T-3 | R-2: PG-19 32K dense = **10.340 has no source run**; closest match is chunk-10 intermediate `10.34` in the 20-chunk rerun. Either reread the rerun's final `9.5712` → Δ=−0.002 still holds, OR locate the 10-chunk run that produced 10.340. | §8.2 L713-714 |
| T-4 | R-1: Paper Tab. 1 rounds 32K Δppl to `−0.002`, but arxiv_v1 cell 22 shows `−0.0042`. Either change Tab. 1 to cite the 20-chunk rerun (−0.0017) consistently with Tab. 3, or update to arxiv_v1's `−0.004`. | Tab. 1 L691 |
| T-5 | A-10 / R-10: either remove the boundary-verification claim and "0 triggers" line, or implement Eq. 30 (see 7.3). | §6.1 L614-621, §8.4 L922 |
| T-6 | C-1 model: either switch paper to cite "Llama 3.1-8B (base)" (matches what was measured) or re-run with Instruct (see 7.3). | §7 L659 |
| T-7 | K-1/K-3/K-5: rewrite §2.3 to describe symmetric scale-only quantisation; drop zero-point from scoring formula §5.2 L572-578. Theorem 2 still holds but needs restated derivation. | §2.3, §5.2 |
| T-8 | V-2: paper narrative must acknowledge that **the headline benchmarks ran with FP16 values, not INT4**. Move INT4 to optional ablation or future work, OR run INT4 ablation (see 7.3). | §3.1, §7 L667 |

### 7.2 Adding an INT4 ablation (~2-4h wall)

If INT4 is to remain a main-paper claim, minimum viable ablation: run `pg19_perplexity.py` + `niah.py` with `--use-int4-values` (new flag needed — half-day to wire) at one or two contexts (8K and 32K) with 5 chunks each. ~2h GPU after wiring. This populates E_val telemetry.

### 7.3 Full re-runs required only if paper text stays as-is on:

- **Model Instruct vs base** (C-1): needs HF token, re-run all 24 arxiv_v1 cells + RULER paper + NIAH 100-trial + overnight rerun. ~2-3 days.
- **INT4 values primary** (V-2): wiring + full re-sweep. ~3-5 days with validation.
- **Asymmetric keys** (K-1): engineering change to kernel, new Δ derivation, new bounds constants, full re-sweep. ~1 week.
- **Boundary verification Eq. 30** (A-10): implement the tail-block log-mass upper-bound check in the kernel + telemetry; re-run to verify "0 triggers" claim. ~1-2 days + re-run.

## 8. My recommendation

Text-only path (7.1) is ~1 day of paper edits and leaves the algorithmic contribution intact — the bounds hold for symmetric keys, the FP16-values path IS what was measured at scale, and the Δppl < 0.01 + RULER parity claims are solid under the measured configuration. Add a short INT4 ablation (7.2) to cover the value-compression claim with a small empirical footprint. Keep Instruct and Eq. 30 as "future work" / limitations.

## 9. What I still owe verification on

- §9 Appendix A (TV bound proof) — mathematical proof, not code.
- §10 Appendix B (NIAH precision ablation) — need to check `benchmarks/results/` for a per-component decomposition artifact.
- §8.5 Storage analysis numbers (Tab. 7, Tab. 8): check `benchmarks/results/` for memory-accounting logs.
- §8.6 Runtime performance (Tab. 9+): already covered by the context-scaling rerun at `benchmarks/results/context_scaling_rerun_20260423/` but I haven't cross-referenced every table cell.

I will not claim these match until I have read the corresponding artifacts. Point me at anything above you want verified first.
