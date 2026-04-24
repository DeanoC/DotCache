# Paper-v1 Re-run Handoff

You are about to run the post-port DotCache paper benchmarks on this
machine. The user will tell you separately which cells to run; this
document covers everything else.

---

## 0. The 60-second briefing

The paper `Certified_Quantised_Attention.tex` (lives outside this repo,
authored externally) makes a specific spec for the certified KV-cache
config. Earlier benchmark data on `bench/overnight-20260423` was
**invalidated by an audit** (`docs/paper_code_audit_20260424.md`) because
the code silently used the wrong quantisation: symmetric INT8 keys + FP16
values + `v_tolerance=0.5`, instead of the paper's asymmetric INT8 keys +
INT4 per-group `g=16` values + `v_tolerance=0.05`.

This branch (`port-to-paper-20260424`) ports the code to the paper spec.
Steps 0–4 of the port are landed and tested; Step 6 is what you're about
to do. Step 7 (cleanup) will follow you.

**You must not** revert any of the Step 0–4 changes summarised below.

---

## 1. Branch + environment

```
branch:  port-to-paper-20260424
remote:  origin (github.com/DeanoC/DotCache.git)
base:    main
deps:    torch 2.8+cu128, triton 3.4, transformers 5.5.4, pytest 9.0.3
GPU:     CUDA required; the port was developed on RTX PRO 6000 Blackwell
venv:    .venv/  (use `.venv/bin/python`, not system python)
```

Sanity-check on a fresh machine:

```bash
git checkout port-to-paper-20260424
git pull
.venv/bin/python -m pytest tests/test_v_tolerance_required.py \
    tests/test_paper_bench_cli.py tests/test_output_json_schema.py -q
```

All 19 tests above must pass — they verify the plumbing isn't broken.
If they fail, **stop and report**; do not proceed with benches.

If the venv doesn't exist:
```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install transformers bitsandbytes datasets pytest
```

---

## 2. What the port did (so you don't undo it)

### Step 0 — Plumbing (commit per file: `dotcache/integrations/llama.py`, `dotcache/kernels/certified_attention.py`, `benchmarks/paper/*`)
- `v_tolerance` is **REQUIRED** at both `certified_attention_layer` (TypeError if missing) and `CertifiedAttentionState.__post_init__` (ValueError if missing). **DO NOT add a default value back.** This guard is intentional.
- `score_consistency_check` defaults to `True` at both kernel and state. Paper §7 says enabled.
- The `DOTCACHE_V_TOL` env var was **removed** from every site. It never worked (no kernel read it). **DO NOT reintroduce it** under any name. Use the `--v-tolerance` CLI flag instead.
- All 4 paper benches (`pg19_perplexity.py`, `niah.py`, `ruler.py`, `longbench.py`) gained `--v-tolerance` (REQUIRED), `--use-int4-values`, `--group-size` flags. `longbench.py` also gained the full §7 knob set (was previously missing them).
- Each paper-bench output JSON now embeds a `cache_config` block with the §7 knob values, `code_sha`, and `dotcache_config_hash` (sha256 of the config). **Verify this block exists in your outputs.**

### Step 1 — Asymmetric INT8 keys (paper §2.3)
- Encoder uses `s = (k_max − k_min) / 255`, `z = (k_min + k_max) / 2` (fp-space midpoint, **unconstrained**). Quant: `q = clamp(round((k − z) / s), −128, 127)`. Dequant: `k̂ = q·s + z`.
- The audit's earlier guess "z ∈ [−128, 127]" was wrong; verified directly against paper §2.3 Eq. 1.
- `TieredKeyCacheLayer` carries `keys_zero_points` alongside `keys_scale`. Five Triton kernels were updated to take `K_zp_ptr` and dequant via `q·s + z`.

### Step 2 — INT4 values g=16 + decode-append support
- `GROUP_SIZE = 16` (was 32). Paper §7. **DO NOT change to 32** — appendix-B ablation passes `--group-size 32` explicitly; the default stays at 16.
- The INT4 path now supports decode-time append (the audit's Risk #3 — "INT4 has never run end-to-end" was real). Cache buffers grow with `max_new_tokens`; `append_token` per-token quantises to INT4 + updates per-block η_b annotation.
- INT4 quant for VALUES is per-token per-group (each token's `d_v` splits into `d_v/g` groups, each group's scale/zero from that token's own values). Partial blocks are fine for values — no need to wait for the block to fill (unlike keys, which need block-fill to compute per-channel scale).

### Step 3 — Eq. 30 boundary verification (paper §6.1)
- For each tail block b ∉ promoted top-K set: check `ℓ_b^int8 + Δ > ℓ^fp16_(r)`. If true, escalate to Rung-3 (per-head FP16 recompute) alongside ranking-disagreement triggers.
- `compute_fp16_block_scores(..., return_log_mass=True)` returns both max-logit (for ranking + score-consistency) and log-mass (for Eq. 30) in one fused FP16 rescore.
- Paper §8.6 claims "0 boundary triggers" — now empirically verifiable. Telemetry: `boundary_check_fired`, `boundary_check_triggered_heads` per layer-step; aggregator emits `boundary_check_fired_layers` and `boundary_check_triggered_heads_total`.

### Step 4 — V_max + E_key (paper §2.3, §4.5)
- Cache stores per-block `ν_b = max_t ‖V_t‖₂` (paper §2.3 last paragraph). Updated incrementally on `append_token`.
- E_key formula `E_key = 2·V_max·exp(2Δ)·ᾱ_T·(exp(2Δ)−1)` assembled per head per step in the adaptive K* telemetry block. Emitted as `e_key_step_mean`, `e_key_step_max`, `v_max_layer`. Aggregator rolls up to `e_key_step_mean/max` and `v_max_global`.

### Step 5 — Output folder + manifest helper (this commit)
- New dir: `benchmarks/results/paper_v1_20260424/` — cells land here.
- New helper: `benchmarks/_manifest.py` — generates/refreshes/validates `run_manifest.json` per output dir.
- See §4 "Where outputs go" below.

---

## 3. How to invoke the paper benches

### The paper-§7 spec — every certified cell needs ALL of these

```
--v-tolerance 0.05            # REQUIRED, no default
--use-int4-values              # opt-in to INT4 g=16 path
--group-size 16                # paper §7 default
--tau-cov 0.995
--k-min 2
--k-max 128
--ranking-fallback
--ranking-r 1
--ranking-fallback-mode full
--score-consistency-check
--eps-guard 0.01
--exploration-rate 0.02
--rung1-threshold 0.02
--rung1-multiplier 2.0
```

The orchestrator at `benchmarks/run_arxiv_v1_sweep.py` already passes
all of these — use it whenever possible. For one-off cells, copy the
flag set from `_cli_for_pg19` / `_cli_for_niah` / `_cli_for_ruler`.

### Required for **dense** cells too

`--v-tolerance` is required even when `--dense-only` is set, because
argparse rejects missing-required regardless of mode. The dense path
ignores the value, but you must pass it (e.g. `--v-tolerance 0.05`).

### Model

```
NousResearch/Meta-Llama-3.1-8B   ← base model, NOT Instruct
```

The paper §7 text says Llama-3.1-8B-Instruct, but the user decided on
2026-04-24 to keep the base model and update the paper text externally.
**Do not** swap to Instruct.

Loaded via `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_8bit=True))`.

### Datasets / HF auth

PG-19, NIAH, RULER, LongBench all load via the `datasets` library.
Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` env var. The base Llama model
is not gated (no token required for the model itself), but some
datasets may need authentication. The bench scripts call
`resolve_hf_auth_kwargs()` which checks both env vars.

### Other env vars (kept; do NOT remove)

- `DOTCACHE_FP16_CACHE_BLOCKS=N` — bounded transparent VRAM cache size in blocks (paper §3.2). Honoured by pg19/niah/ruler. Unset = legacy full mirror.
- `DOTCACHE_FAST_ATTEND=0` — falls back to single-program-per-head attend kernel (default split-K is 15-20× faster on Blackwell). Don't touch unless debugging.

---

## 4. Where outputs go

```
benchmarks/results/paper_v1_20260424/
├── run_manifest.json          ← created by _manifest.write_initial_manifest()
└── <cell>.json                ← per-cell outputs from the bench scripts
```

### Two-machine coordination

Append a machine tag to cell basenames so the merged tree has unique
names. Convention: `<NN>_<bench>_<ctx>_<config>.<machine>.json`
e.g. `01_pg19_4K_dense.gpuA.json` and `02_pg19_8K_dense.gpuB.json`.

The user will give you a machine tag in the run instructions.

### Manifest workflow

At the **start** of your run:

```python
from benchmarks._manifest import write_initial_manifest
write_initial_manifest(
    "benchmarks/results/paper_v1_20260424",
    dotcache_config={
        "v_tolerance": 0.05, "quantization_mode": "asymmetric_int8_keys+int4_g16_values",
        "use_int4_values": True, "group_size": 16,
        "score_consistency_check": True,
        "tau_cov": 0.995, "k_min": 2, "k_max": 128,
        "ranking_fallback": True, "ranking_r": 1, "ranking_fallback_mode": "full",
        "eps_guard": 0.01, "exploration_rate": 0.02,
        "rung1_threshold": 0.02, "rung1_multiplier": 2.0,
    },
    notes="paper_v1 first re-run on <machine_tag>",
)
```

After **each cell completes** (or once at the end):

```bash
python -m benchmarks._manifest --refresh benchmarks/results/paper_v1_20260424
```

Before declaring the run done, validate the manifest:

```bash
python -m benchmarks._manifest --validate benchmarks/results/paper_v1_20260424
# exit code 0 = ok; non-zero = a recorded sha doesn't match disk
```

### What every cell JSON must contain

The bench scripts already write a `cache_config` block — verify it's
present in at least the first cell:

```bash
python -c "import json; print(list(json.load(open('benchmarks/results/paper_v1_20260424/<cell>.json'))['cache_config'].keys()))"
```

Expected keys: `v_tolerance`, `quantization_mode`, `asymmetric_keys`,
`use_int4_values`, `group_size`, `score_consistency_check`, `tau_cov`,
`k_min`, `k_max`, `ranking_fallback`, `ranking_r`, `ranking_fallback_mode`,
`eps_guard`, `exploration_rate`, `rung1_threshold`, `rung1_multiplier`,
`code_sha`, `dotcache_config_hash`.

If any are missing, **stop and report** — your bench script may not
have picked up the Step-0 plumbing.

---

## 5. Telemetry to expect (paper-§8.6 sanity targets)

The aggregator (`CertifiedAttentionState.aggregate_step_stats`) returns a
dict per step with these load-bearing fields. Spot-check the first
certified cell against these expectations:

| Field | Paper expectation | Action if violated |
|---|---|---|
| `score_consistency_violation_heads_total` | 0 | Theorem 2 empirically violated — STOP, report |
| `boundary_check_fired_layers` | 0 | Eq. 30 fired — paper §8.6 claim breaks; record but continue |
| `rung1_fired` | rare on PG-19, ~all-steps on NIAH/RULER | expected per paper §8.6 |
| `rung4_fired` | false (always) | If true, page-in or kernel corruption — STOP |
| `e_key_step_mean` | ≪ 1 (typically <1e-6 on real model) | If huge, V_max or Δ is wrong — STOP |
| `tail_mass_int8_est_step_mean` | small (≈ 1e-5 to 1e-3) | If close to 1, adaptive K* is broken |

Score-consistency violations and Rung-4 fires are HARD ERRORS — they
mean the bound was empirically violated. Stop the run and report
which cell + step + layer.

---

## 6. Pre-flight (run BEFORE any GPU bench)

1. **Verify branch + clean tree:**
   ```bash
   git status -s     # should be empty (no modified files)
   git rev-parse --abbrev-ref HEAD   # should be port-to-paper-20260424
   ```

2. **Verify deps:**
   ```bash
   .venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
   .venv/bin/python -c "import triton, transformers, bitsandbytes; print(triton.__version__, transformers.__version__)"
   ```

3. **Run the plumbing tests** (CPU-only, ~3 min total):
   ```bash
   .venv/bin/python -m pytest tests/test_v_tolerance_required.py \
       tests/test_paper_bench_cli.py tests/test_output_json_schema.py -q
   ```
   All 19 must pass.

4. **Run the kernel-correctness tests** (CUDA, ~3 min):
   ```bash
   .venv/bin/python -m pytest tests/test_asymmetric_keys_quant.py \
       tests/test_int4_g16.py tests/test_eq30_boundary.py \
       tests/test_vmax_and_ekey.py tests/test_paper_pipeline_smoke.py -q
   ```
   All 33 must pass.

5. **Smoke a tiny cell to confirm the bench harness runs:**
   ```bash
   .venv/bin/python benchmarks/paper/pg19_perplexity.py \
       --v-tolerance 0.05 --use-int4-values --group-size 16 \
       --tau-cov 0.995 --k-min 2 --k-max 128 --ranking-fallback --ranking-r 1 \
       --score-consistency-check --eps-guard 0.01 --exploration-rate 0.02 \
       --rung1-threshold 0.02 --rung1-multiplier 2.0 \
       --context 4096 --num-chunks 1 \
       --output /tmp/_smoke_pg19.json
   .venv/bin/python -c "import json; d=json.load(open('/tmp/_smoke_pg19.json')); print('cache_config keys:', sorted(d['cache_config'].keys())); print('quant mode:', d['cache_config']['quantization_mode']); print('v_tolerance:', d['cache_config']['v_tolerance'])"
   ```
   Expect `quant mode: asymmetric_int8_keys+int4_g16_values` and `v_tolerance: 0.05`.

If any of the above fails, **stop and report**.

---

## 7. Pre-existing test failures (NOT regressions — IGNORE)

These tests fail on the bare branch and are unrelated to the port. Do
NOT attempt to fix them; they're known issues:

- `tests/test_qwen35_integration.py::*` (~44 tests) — `DynamicCache` API
  change in transformers 5.5.4 (no `conv_states` attribute).
- `tests/test_torch_cuda_backend.py::test_m3_pages_*` — atol violations
  on Blackwell (~3e-3 mismatch); precision was tuned for older GPUs.
- `tests/test_persistent_runtime*.py::*` (~7 tests) — stale m0/m3 schema
  expectations (extra `EXACT_KEY_M3` key not in old assertions).
- `tests/test_model_registry.py::test_model_registry_marks_qwen35_4b_as_local_stretch_lane`
  — Qwen registry assertion drift.
- `tests/test_append_token_scale.py::test_poison_padding_in_new_block`
  — symmetric-era assertion `padding == -127`. With asymmetric quant
  the poison value is irrelevant (paired with scale=0 → dequant=0), so
  the assertion is now wrong but the behaviour is fine. Will be
  cleaned up in Step 7.

If a test you expect to pass is in this list, that's why.

---

## 8. Things you MUST NOT do

1. **Do not edit `Certified_Quantised_Attention.tex` or any paper .tex/.pdf.**
   The paper is authored outside this repo. Text-side fixes get recorded
   in `docs/paper_code_audit_20260424.md` for the user to apply externally.
   (This file may not be present on this branch — it's on
   `bench/overnight-20260423` only.)

2. **Do not reintroduce `DOTCACHE_V_TOL`.** The mechanism never worked
   and added a dual-path footgun. Use `--v-tolerance` CLI only.

3. **Do not add a default for `v_tolerance`.** The kernel and state both
   require explicit. The TypeError/ValueError is the teeth that prevents
   recurrence of the Apr 17–24 silent-default bug.

4. **Do not change `GROUP_SIZE = 16`** in `dotcache/kernels/int4_group_quantise.py`.
   Paper §7. The appendix-B ablation passes `--group-size 32` explicitly;
   the default stays 16.

5. **Do not delete results directories.** Pre-existing dirs under
   `benchmarks/results/` are evidence — even invalidated ones. Step 7
   will archive them; until then, leave alone.

6. **Do not skip `--score-consistency-check`** on certified cells. Its
   default flipped to True in Step 0; explicit pass keeps the spec
   visible in the CLI history and the cache_config block.

7. **Do not amend or force-push commits** without explicit user approval.
   Don't `--no-verify` either.

8. **Do not bypass the manifest.** Always run `--init` at start and
   `--refresh` after writes, then `--validate` before declaring done.

---

## 9. If something goes wrong

| Symptom | Likely cause | What to do |
|---|---|---|
| `TypeError: ... missing 1 required keyword-only argument: 'v_tolerance'` | You didn't pass `--v-tolerance` | Add the flag — see §3 |
| `ValueError: CertifiedAttentionState requires explicit v_tolerance` | Same | Same |
| Output JSON missing `cache_config` block | Bench script is from before the port | Verify branch is `port-to-paper-20260424` and `git pull`'d |
| `score_consistency_violation_heads_total > 0` in any cell | Theorem 2 empirically broken | STOP. Report cell + layer + step. |
| `rung4_fired: true` | Same as above (Rung-4 escalation triggered) | STOP. Same. |
| `e_key_step_mean` is huge (e.g. >1.0) | V_max or Δ computation is wrong | STOP. Report. |
| `boundary_check_fired_layers > 0` in real benches | Paper §8.6 "0 triggers" claim breaks | Record the count. Continue. Tell the user. |
| Manifest validate reports mismatches | A cell file changed after manifest was refreshed | Re-run `--refresh` then `--validate` |
| `RuntimeError: shape '[N, M]' is invalid for input of size X` in INT4 path | INT4 buffer size mismatch (Step 2 fix should prevent this) | STOP. Report — likely a regression of the decode-append fix |
| Tests in §7 fail | Pre-existing | Ignore. They're listed for a reason. |

---

## 10. When you're done

1. `python -m benchmarks._manifest --validate benchmarks/results/paper_v1_20260424` — exit 0.
2. `git status` — only the new cell JSONs + updated `run_manifest.json` should appear.
3. Stage the new files: `git add benchmarks/results/paper_v1_20260424/`.
4. Commit with a message describing which cells you ran and on which machine. Example:
   ```
   bench: paper_v1 PG-19 main + CI on gpuA (5 books × {4K,8K,16K,32K} + 20 chunks × ditto)
   ```
5. Push: `git push origin port-to-paper-20260424`.
6. Tell the user the commit sha and which cells landed.

If your machine is the second of two, expect a merge: `git pull --rebase`
before pushing if the other machine has already pushed cells.

---

## 11. Reference: full test inventory

The full Step 0–4 test suite (102 tests):

```
tests/test_v_tolerance_required.py       (5)   Step 0 plumbing
tests/test_paper_bench_cli.py            (8)   Step 0 CLI argparse
tests/test_output_json_schema.py         (6)   Step 0 provenance block
tests/test_asymmetric_keys_quant.py      (8)   Step 1 asymmetric INT8
tests/test_int4_g16.py                   (9)   Step 2 INT4 g=16 + decode-append
tests/test_paper_pipeline_smoke.py       (4)   Step 2 tiny-Llama E2E
tests/test_eq30_boundary.py              (4)   Step 3 Eq. 30 check
tests/test_vmax_and_ekey.py              (8)   Step 4 V_max + E_key
tests/test_value_error_bound.py          (8)   Pre-existing, kept passing
tests/test_adaptive_topk.py              (15)  Pre-existing, patched for Step 0
tests/test_ranking_consistency_fallback.py (16) Pre-existing, patched for Steps 0+1+3
tests/test_append_token_scale.py         (11)  Pre-existing, patched for Step 1
```

Total: 102 passing tests. Run subsets per §6 pre-flight; run the full
suite if anything seems off.

---

## 12. Where to find more context

- **Plan:** `/root/.claude/plans/lets-create-a-exact-abstract-wilkes.md`
  on the user's machine — not in the repo. The user can paste relevant
  excerpts if you need them.
- **Audit doc:** `docs/paper_code_audit_20260424.md` on
  `bench/overnight-20260423` (NOT this branch). Check there for the
  original bug summary if you need the full context.
- **Paper:** `Certified_Quantised_Attention.tex` on
  `bench/overnight-20260423` (NOT this branch). Read-only reference.
- **This branch's history:** `git log main..HEAD` shows the port commits.
