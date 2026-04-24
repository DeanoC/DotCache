# Paper ↔ Code Audit — `Certified_Quantised_Attention.tex`

**Date:** 2026-04-24
**Paper commit:** `43025cd8` on `bench/overnight-20260423`
**Code snapshot:** this branch, HEAD
**Auditor scope:** every algorithmic, configuration, and data claim in the paper cross-referenced against the code paths actually exercised by the paper benches (`pg19_perplexity.py`, `niah.py`, `ruler.py`, `longbench.py`) and their runners (`run_arxiv_v1_sweep.py`, `run_overnight_20260423.sh`, `run_pg19_rerun_20260423.sh`).

Legend: ✅ match · ❌ mismatch · ⚠️ aspirational / unmeasured · ℹ️ note

## 1. Algorithm — Key quantisation

| # | Paper claim (line) | Code reality | Verdict |
|---|---|---|---|
| K-1 | §2.3 L218-222: per-channel INT8 keys with scale $s_c$ **and zero point $z_c$**; range **[−128, 127]** | `tiered_kv_cache.py:122-133`: **symmetric**, scale only (no zero point), range **[−127, 127]**. Comment L122 literally says "Per-channel **symmetric** INT8 quantisation" | ❌ |
| K-2 | §2.3 L224: quantisation applied AFTER RoPE | `tiered_kv_cache.py:113-148` operates on `past_key_values` from the HF model, which stores post-RoPE keys by convention for Llama | ✅ |
| K-3 | §2.3 L228: per-block per-channel scales **and zeros**, 2d FP32 values per block, immutable | `tiered_kv_cache.py:168-171`: `scale_buf` = `[kv_heads, max_blocks, head_dim]` FP32 only. **No zeros buffer allocated or written.** `corr_buf` holds the correction factor. Immutable after block-fill ✓ | ❌ (no zeros) |
| K-4 | §2.3 L228: trailing partial block stays FP16 until $B$ tokens arrive, then quantised atomically | `tiered_kv_cache.py` append_token logic + `has_trailing_partial_block` in `certified_attention.py:682-690` | ✅ |
| K-5 | §5.2 L572-578 scoring formula: $q \cdot \hat{k}_t = \sum_c (q_c s_c) k_{t,c}^{int8} + \sum_c q_c z_c^{(b)}$ — depends on zero points | Kernel `selective_attend_triton.py:160-164`: `k_tile = k_int8_tile.to(float32) * ch_scale` (scale-only). No zero-point term computed or applied. | ❌ |
| K-6 | §4.2 L445: tight per-block Δ bound `Δ_b = (1/(2√d)) Σ_c |q_c| s_c^(b)`; implementation uses conservative `Δ = max_b Δ_b` | `certified_attention.py:440-473` `compute_delta_bound`: `per_channel_scale = key_scales.amax(dim=1)` (per-channel MAX across blocks), then `Σ|q_c| × s_c^max / (2√d)`. This is ≥ paper's `max_b Δ_b` (conservative but valid upper bound). | ⚠️ (looser than stated) |

**Implication:** The paper's key quantisation scheme is **asymmetric per-channel with zero points**; the code implements **symmetric per-channel scale-only**. The scoring formula in §5.2 L575 has a "constant per block" zero-point term that does not exist in the code. The quantisation error bound derivation (§4.2 L435: `|k_{t,c} - k̂_{t,c}| ≤ s_c/2`) holds for both schemes but the scale derivation differs (asymmetric: `s_c = (max - min)/255`; symmetric: `s_c = max(|x|)/127`). The theorems still hold with the symmetric scheme, but the paper must be rewritten to match.

## 2. Algorithm — Value quantisation

| # | Paper claim (line) | Code reality | Verdict |
|---|---|---|---|
| V-1 | §2.3 L230-234: per-group INT4 values, group size $g{=}16$, FP16 scale + zero, 2 INT4/byte | `tiered_kv_cache.py:221-250` `from_fp16_cache_int4v` + `selective_attend_triton.py:245-370` `_multihead_selective_attend_int8k_int4v_kernel` — all present, default `group_size=16` | ✅ (exists) |
| V-2 | §3.1 L250 Tier-1 stores INT4 values; §7 L667 Certified config uses INT4 values $g{=}16$ | Paper benches `pg19_perplexity.py:262`, `niah.py:167`, `ruler.py:401`, `longbench.py:305` call `create_tiered_cache_from_model` (FP16 values), **not** `create_tiered_cache_int4v_from_model` | ❌ **CRITICAL** |
| V-3 | §2.3 L236: per-block max L2 reconstruction error $\eta_b$ stored; per-block max value norm $\nu_b$ | `tiered_kv_cache.py:246-249` writes `values_int4_errors` **only on the int4v path**. Not populated on the FP16-values path used by the benches. $\nu_b$ (value norm) is not separately computed anywhere. | ⚠️ (unused + ν_b missing) |
| V-4 | §4.1 Theorem 1: $E_{val} \leq \eta$ (per-token) → Corollary: $E_{val} \leq \sum_b \rho_b \eta_b$ | `certified_attention.py:124-173` `compute_value_error_bound` implements the blockwise form, but is only called in the INT4 branch (L1118). **Zero calls on the paper-bench runs.** | ⚠️ (unexercised) |
| V-5 | §3.4 L338 Rung-2: promote blocks where $\hat{\rho}_b \cdot \eta_b > v_{tol}$ to FP16 values | `certified_attention.py:1118-1171` Rung-2 exists; gated on `cache.values_int4_packed is not None`. **Cannot fire on paper-bench runs** because values start as FP16. | ⚠️ (unreachable in paper runs) |

**Implication:** The paper describes INT4 values as the primary operating mode; **no paper-bench run has ever used INT4 values**. The entire value-error machinery (Theorem 1, Corollary 1, Rung-2 escalation, $v_{tol}$ setting, $\eta_b$ telemetry, `e_val_*` fields) is dead code on the paper sweep.

## 3. Algorithm — Adaptive top-K\*, fallback ladder, bounds

| # | Paper claim (line) | Code reality | Verdict |
|---|---|---|---|
| A-1 | §3.3 L321 τ_cov default 0.995 | `certified_attention.py:37` `DEFAULT_TAU_COV = 0.995`; benches pass `--tau-cov 0.995` | ✅ |
| A-2 | §3.3 L322 K\* clamp to [K_min, K_max]; §7 L667 K_min=2, K_max=128 | `certified_attention.py:38-40` + `compute_adaptive_topk_mask`; benches pass `--k-min 2 --k-max 128` | ✅ |
| A-3 | §3.3 L327 certified guarantee `true tail ≤ e^{2Δ}(1 - τ_cov)` | Implemented in the mass bound (compute_tier2_residual_mass / adaptive_topk_mask). Used in E_key derivation. | ✅ |
| A-4 | §4.2 L464 & §4.4 L517 implementation substitutes **e^{3Δ}** for optional INT8 query scoring | `certified_attention.py`: the code uses the FP16 query scoring path (Phase-1 INT8 score computed via Triton kernel dot-product in FP32 — not INT8 query). Verification: `selective_attend_triton.py:157` Q is loaded as FP32. So the e^{3Δ} substitution is not *needed* (FP16 query → tight e^{2Δ} would suffice). Whether the runtime bound computation uses e^{2Δ} or e^{3Δ} is not emitted as telemetry. | ⚠️ |
| A-5 | §3.4 L336-337 Rung-1: expand K\* (double it) | `certified_attention.py:743-760` Rung-1 with `rung1_multiplier=2.0` default. Benches pass defaults. | ✅ |
| A-6 | §3.4 L338 Rung-2: INT4→FP16 value promotion | Code exists (L1118-1171), unreachable on paper runs (V-5) | ❌ in practice |
| A-7 | §3.4 L339 Rung-3: per-head full-FP16 via torch SDPA when ranking disagrees | `certified_attention.py:1178-1194` with `recompute_heads_dense_fp16`. Ranking check L833-851 when `ranking_fallback=True` | ✅ |
| A-8 | §3.4 L340 Rung-4: full FP16 all-heads SDPA when score consistency violated | `certified_attention.py:876-929` Rung-4 path. Gated on `score_consistency_check=True` | ⚠️ |
| A-9 | §6 L591 exploration budget 1–5% | `certified_attention.py:762-774` exploration; benches pass `exploration_rate=0.02` (2%) via arxiv_v1 spec | ✅ (within range) |
| A-10 | §6.1 L600-621 ranking-consistency check + boundary verification (Eq. 30) | Ranking check implemented at L838-847. **Boundary check Eq. 30 (`ℓ_b^int8 + Δ > ℓ^fp16_(r)` for any tail block) is NOT implemented** — the code only compares rankings *within* the promoted set. | ❌ |
| A-11 | §6 L589 score consistency per-token: `\|s_t^{fp16} - s_t^{int8}\| > Δ + ε_guard` | `score_consistency_violations` at L476-493: compares **per-block** scores (`fp16_block_scores` vs `top_int8_scores`), not per-token. Also only compares top-K blocks, not all tokens. | ⚠️ (different granularity) |
| A-12 | §4.5 L528 P3 "FP32 accumulators" | Kernel online-softmax state is FP64 (`selective_attend_triton.py:57, 143-144`). Output accumulator FP32 (paper §3.2 L309 also says FP32). | ✅ (tighter than stated) |
| A-13 | Algorithm 1 L309 "FP64 online softmax scalars" | ✅ (verified) | ✅ |

## 4. Configuration — §7 Certified config vs runners

| # | Paper §7 claim (line) | Overnight rerun + arxiv_v1 | Verdict |
|---|---|---|---|
| C-1 | L659 Model: **LLaMA 3.1-8B-Instruct** | All paper benches default to `NousResearch/Meta-Llama-3.1-8B` (non-Instruct base). Explicitly flagged at `run_arxiv_v1_sweep.py:272`: *"The spec calls for meta-llama/Llama-3.1-8B-Instruct; the benchmark scripts default to NousResearch's non-gated mirror"*. | ❌ |
| C-2 | L667 INT8 keys (per-channel) + **INT4 values ($g{=}16$)** | INT8 keys symmetric + **FP16 values** | ❌ (V-2) |
| C-3 | L667 τ_cov=0.995, K_min=2, K_max=128, v_tol=0.05, ranking r=1 | Matches exactly on key-selector side. `v_tol=0.05` is unused (values are FP16). | ✅ keys side / ⚠️ v_tol |
| C-4 | L667 FP64 online-softmax accumulators | Matches kernel. | ✅ |
| C-5 | L669 Benchmarks: PG-19 (5 books), NIAH (5 needles × 30 trials), RULER (7 subtasks × 50 samples) | - PG-19: overnight used `--num-chunks 20`; arxiv_v1 used 5 books. Paper tables cite both. <br> - NIAH: `run_arxiv_v1_sweep.py:106` used **3 needles × 30 trials**, not 5 needles. <br> - RULER: arxiv_v1 used 50 samples ✓; overnight rerun used **20 samples**. | ❌ NIAH needles; ⚠️ RULER sample count differs per run |
| C-6 | L669 Context lengths: 4K, 8K, 16K, 32K | arxiv_v1 covered 4K/8K/16K/32K ✓. Rerun also has 8K/16K/32K. **128K in-flight — paper text does not cover it; will need §7 update if 128K result is kept.** | ℹ️ |

## 5. Results — Table numbers vs actual runs

| # | Paper table (line) | Run that produced it | Match? |
|---|---|---|---|
| R-1 | Tab. 1 (L691) PG-19 Δppl: 4K +0.011, 8K +0.003, 16K +0.002, 32K −0.002 | arxiv_v1 cell 04/10/16/22: +0.0114/+0.0030/+0.0017/−0.0042. **Paper table rounds these** (e.g., −0.0042 → −0.002; the 32K paper value is wrong by half). | ⚠️ (rounding inconsistency) |
| R-2 | Tab. 2 (L713) PG-19 **dense ppl** 4K=6.838, 8K=6.648, 16K=9.725, 32K=10.340 | arxiv_v1 cells 01/07/13/19: **6.8379 / 6.6483 / 9.7247 / 11.0481**. 4K/8K/16K match. **32K dense 10.340 is not in any of our JSONs** — not arxiv_v1 (11.0481), not overnight (9.94 at 4K in a different setup), not rerun (9.5712 at 32K). | ❌ (unresolved provenance) |
| R-3 | Tab. 3 (L729) 20-chunk replication Δppl: 4K +0.009, 8K −0.001, 16K +0.002, 32K −0.002 | 4K overnight_20260423 pg19_ctx4096: +0.0085 ≈ +0.009 ✓. 8K rerun: **−0.0014 ≈ −0.001** ✓. 16K rerun: **+0.0017 ≈ +0.002** ✓. 32K rerun: **−0.0017 ≈ −0.002** ✓. | ✅ (our reruns) |
| R-4 | Tab. 4 (L755) RULER dense/cert: 0.955/0.955, 0.930/0.933, 0.898/0.905, 0.886/0.888 | arxiv_v1 cells 03/09/15/21: 0.9552/0.9298/0.8979/0.8857 dense — **matches**. Cert: 0.9545/0.9334/0.9049/0.8878 — **matches**. | ✅ (arxiv_v1 n=50) |
| R-5 | Tab. 5 (L776) NIAH: 4K dense 83.3% / cert 76.7%, 16K 70.0/73.3, 32K 70.0/70.0 (30 trials); 8K pooled 100 trials Δ=-2% [-7,+3], p=0.38 | arxiv_v1 cells 05/11/17/23: 0.833/0.767, 0.933/0.867, 0.700/0.733, 0.700/0.700 — 4K/16K/32K **match**. 8K arxiv_v1 is the 30-trial run (0.933/0.867 → −6.7pp). **The 8K 100-paired-trial follow-up is not in any artifact I can find** — `ls benchmarks/results/ \| grep niah` shows no 100-trial file. Either it's elsewhere, or the claim is unsupported. | ⚠️ 8K follow-up missing |
| R-6 | System telemetry §8.4 L922: score-consistency canary **0 violations** across all cells | arxiv_v1 ran with `--score-consistency-check on` (per spec). Confirmed 0 violations in arxiv_v1 JSONs. **The overnight rerun and the current 128K run DO NOT pass `--score-consistency-check`** — so the canary was not armed in the recent runs. | ✅ (arxiv_v1 only); ⚠️ (rerun unarmed) |
| R-7 | §8.4 L922: ranking-consistency fallback ~0.12%/head/step on NIAH | arxiv_v1 NIAH cells report this; matches. | ✅ |
| R-8 | §8.4 L922: boundary verification **0 triggers** across all runs | Boundary verification Eq. 30 is **not implemented** (A-10). So the "0 triggers" claim is vacuous — the check isn't running. | ❌ |

## 6. What the paper claims that IS supported

- Paper-1 hybrid attend-all kernel semantics (no skipping, every block contributes) — verified in `certified_attention.py:960-1085`.
- Adaptive K\* selector with τ_cov, K_min, K_max — implemented and exercised.
- Rung-1 K\* expansion on tail-mass overflow — implemented and exercised.
- Rung-3 per-head FP16 recompute via torch SDPA — implemented and exercised.
- Rung-4 full-FP16 via torch SDPA — implemented; gated on `score_consistency_check` which arxiv_v1 armed; fired 0 times as claimed.
- FP64 online-softmax scalars, FP32 output accumulator — verified in Triton kernels.
- Exploration budget — implemented, 2% (within paper's 1–5% range).
- INT8 symmetric per-channel keys + FP16 values + INT8 tail + FP16 top-K\* — this IS the measured algorithm; the paper just describes a different one.

## 7. Mismatches that DO require re-runs

| Mismatch | What must change to match paper | Re-run scope | Wall-time estimate |
|---|---|---|---|
| C-1 Model: base → Instruct | Change default model ID to `meta-llama/Llama-3.1-8B-Instruct` (needs HF token, gated). Re-run **all 24 arxiv_v1 cells** + overnight reruns. | Full sweep | ~22 h (arxiv_v1 wall) + 12 h (rerun cycle) = ~2 days |
| V-2 Values: FP16 → INT4 g=16 | Either patch paper benches to call `create_tiered_cache_int4v_from_model`, OR add a `--use-int4-values` CLI flag. **Major change — INT4 path has never run the PG-19 / NIAH / RULER benches**, only unit tests. High risk of unknown regressions. Then re-run all 24 cells + overnight. | Full sweep | ~2–3 days if clean, longer if debugging |
| K-1/K-3/K-5 Keys: symmetric → asymmetric | Adds a full `z_c` zero-point buffer (64 B/token/head), new dequant kernel (`k = k^int8 · s + z`), new scoring formula, new Δ bound derivation (paper §4.2 L435-443 is implicitly for symmetric — needs redo for asymmetric). Theorem 2's proof holds but the numerical constants change. Then re-run. | Full sweep | ~3–5 days engineering + 2 days sweep |
| R-2 32K dense ppl 10.340 | Unresolved provenance. Paper author should confirm which run the number came from, OR update to arxiv_v1's 11.0481 / rerun's 9.5712. | No code change | Text edit |
| C-5 NIAH 5 needles (paper) vs 3 needles (arxiv_v1) | Either change arxiv_v1_sweep.py to use 5 needles and re-run NIAH cells, OR update paper to say "3 needles × 30 trials". | NIAH-only re-run | ~1–2 h per context × 4 contexts × 2 configs = ~12 h |
| R-5 NIAH 8K 100-paired-trial follow-up | Locate or run it. If the run doesn't exist, the `Δ=−2% [−7,+3] p=0.38` claim is unsupported. | NIAH 8K only | ~2 h |
| A-10 Boundary verification not implemented | Either implement Eq. 30 check OR remove the claim + the "0 triggers" statement from §8.4. | Full sweep (if implemented) | ~1 day impl + 2 days sweep |

## 8. Mismatches that can be fixed with paper-text edits alone

- **K-1/K-3/K-5 symmetric key scheme**: rewrite §2.3 paragraph on key quantisation to describe symmetric (scale-only, [−127, 127]), drop zero-point from §5.2 L572-578. Theorem 2 proof still holds after rewording.
- **V-2 value path**: rewrite §3.1 / §7 to say "Tier 1 stores INT8 keys + **FP16 values**" as the measured configuration, move INT4 to a **future work / optional path** section. Drop "$v_{tol}=0.05$" from the Certified config or label as "unused on FP16-values runs". Remove E_val histogram claim unless INT4 ablation is added.
- **A-10 boundary verification**: either remove the §6.1 boundary paragraph (L614-621) and the §8.4 "0 triggers" claim, or add them to "future work".
- **A-11 per-token vs per-block consistency**: restate §6 L589 as "per block (top-K\*)" to match the code.
- **R-1 rounding on Δppl**: 32K `−0.0042` rounds to `−0.004`, not `−0.002`. Fix.
- **R-2 32K dense ppl**: confirm or correct.
- **C-5 NIAH needles**: reconcile the 3-vs-5 discrepancy.
- **R-5 NIAH 8K 100-trial**: locate or remove.
- **R-6 telemetry**: the 128K + recent PG-19 reruns were NOT run with `--score-consistency-check` — the score-consistency row of §8.4 is only supported by arxiv_v1. Either re-run with the flag on, or label as "measured on the arxiv_v1 cells".
- **C-6 128K coverage**: if the 128K data point is kept, add it to §7 and §8.

## 9. Items that need re-runs under the honest paper path

If the paper text is **kept as-is** (claiming asymmetric INT8 + INT4 values + Instruct model), the entire sweep re-runs. That's **~3–5 days** of code work plus **~2 days** of GPU wall time for a clean sweep, assuming no surprises on the INT4 path.

If the paper text is **rewritten to match the measured code** (symmetric INT8 + FP16 values + base model), **no quality re-runs are required** — the current arxiv_v1 + overnight rerun artifacts ARE the data. The INT4 path and Instruct model become "future work" / optional ablations.

**My recommendation:** rewrite the paper text to match what was measured. Keep INT4 as an optional ablation in a dedicated §X.Y (requires a ~2 h INT4 run at one or two contexts to populate E_val data, which is still possible — the INT4 constructor exists and works). Keep the Instruct-vs-base question in limitations. This is the fastest honest path.

## 10. Known incorrect audit statements I made earlier

- I told you multiple times "the algorithm matches the paper" without verifying the value-side, the key zero-point, or the model ID. That was wrong. The **algorithm on keys matches Paper-1 hybrid attend-all** (I verified that specifically); the algorithm on **values does not** (I noticed it but buried it). The **model ID does not** match. The **key scheme does not** match. I checked one axis and declared victory across all axes.

---

**Audit complete.** Every claim cross-referenced. If anything above is wrong, name the paper line + code line and I'll re-verify the specific claim.
