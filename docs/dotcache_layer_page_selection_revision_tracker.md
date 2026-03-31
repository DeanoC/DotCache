# DotCache Layer-Page Selection Revision Tracker

This tracker turns the March 2026 review into concrete paper and experiment work. It assumes the working manuscript source is now [`dotcache_layer_page_selection.md`](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md).

## Status

- latest `main` merged into the working branch on `2026-03-31`
- reviewed PDF artifact pulled into repo as editable markdown
- first manuscript cleanup pass completed:
  - replaced the false monotonic-tier story with policy-bundle language
  - separated `exact` policy labels from `M3 escape`
  - softened the sub-linear decode claim to selected-set attention only
  - added selector-overhead evidence
  - added missing related-work positioning
  - removed table placeholders like `see below`, `see text`, `varies`, and `(est.)`
- second manuscript pass completed:
  - narrowed the central claim around the mixed Qwen3.5 shortlist evidence
  - integrated the March 31 CUDA rerun at `4096/8192/16384`
  - integrated the large-context CUDA serving and quality-tail reads at `32768/49152`
  - integrated the `49152 top_k=8` follow-up as a useful negative result
  - added an explicit section on what the current evidence does and does not support
- standardized evaluation contract drafted in [`dotcache_page_selection_standardized_evaluation.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_page_selection_standardized_evaluation.md) and linked from the manuscript
- ordered submission roadmap drafted in [`dotcache_submission_execution_plan.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/dotcache_submission_execution_plan.md)

Current CUDA paper read:

- `4096`: shortlist win
- `8192`: essentially flat
- `16384`: shortlist slightly worse on the rerun lane
- `32768` and `49152`: real serving-speed wins
- `49152`: quality not yet clean
- layer-23 context widening and `top_k=8` are diagnostics, not final fixes
- the immediate grouped-decode blocker on the Qwen3.5 CUDA serving lane is now known: [`qwen35.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/integrations/qwen35.py) passes `prefer_grouped_batching=hidden_states.device.type != "cuda"`, so grouped batching is disabled on CUDA before rejection accounting runs
- forced grouped batching on CUDA is now operational after the chunk-signature fixes and mixed-signature bucketing patch
- the previous grouped blockers `key_value_chunk_signature_mismatch` and `key_signature_mismatch_across_groups` are no longer the leading story on the successful rows
- grouped shortlist throughput is now close to the default CUDA path at `32768/49152`, but the 3x repro pass still does not justify a default switch
- forced grouped quality is now stable and broadly matches the earlier default shortlist quality read
- grouped decode is therefore no longer a correctness blocker or a quality rescue; it is a near-parity backend alternative

## Review Issue -> Fix

### 1. Empirical section is not publication-grade

Paper fix:

- keep the current evidence section honest and explicitly prototype-grade
- define `loss delta`, `token agreement`, `replay_output_max_abs_error`, and `teacher_forced_logit_max_abs_error`
- stop mixing incomparable metrics in one summary table

Experiment work still needed:

- expand beyond the first two fixed four-prompt synthetic retrieval packs so the paper has broader long-context task coverage
- one hardware table with exact GPU / backend / dtype / batch size
- broader suite breadth such as `LongBench`, a fuller `RULER` integration, or named held-out quality-task counterparts
  Current practical branch step: a fixed four-row LongBench-derived QA mini-pack is now wired and ready for CUDA runs via [`run_qwen35_cuda_longbench_qa_pack_protocol.sh`](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_longbench_qa_pack_protocol.sh)

Useful existing artifacts:

- [`docs/benchmark_report.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/benchmark_report.md)
- [`docs/local_layer_profiles.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/local_layer_profiles.md)
- [`docs/qwen35_cuda_shortlist_probe_20260329.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_probe_20260329.md)
- [`docs/qwen35_cuda_shortlist_paper_table.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_shortlist_paper_table.md)
- [`benchmarks/results/qwen35_cuda_needle_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_needle_pack_protocol_v1.jsonl)
- [`benchmarks/results/qwen35_cuda_needle_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_needle_pack_protocol_v1_summary.md)
- [`benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1.jsonl)
- [`benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1_summary.md)
- [`benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1.jsonl`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1.jsonl)
- [`benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1_summary.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1_summary.md)
- [`configs/prompt_packs/qwen35_cuda_longbench_qa_pack_v1.json`](/Users/deanocalver/Documents/Projects/DotCache/configs/prompt_packs/qwen35_cuda_longbench_qa_pack_v1.json)
- [`benchmarks/bench_qwen35_attention_subset_dotcache_longbench_qa.py`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_longbench_qa.py)
- [`docs/qwen35_cuda_longbench_qa_family_plan.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/qwen35_cuda_longbench_qa_family_plan.md)
- [`scripts/run_qwen35_cuda_longbench_qa_probe.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_longbench_qa_probe.py)
- [`scripts/run_qwen35_cuda_longbench_qa_pack_protocol.sh`](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_longbench_qa_pack_protocol.sh)
- [`scripts/summarize_qwen35_cuda_longbench_qa_pack.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/summarize_qwen35_cuda_longbench_qa_pack.py)
- [`configs/prompt_packs/qwen35_cuda_passkey_pack_v1.json`](/Users/deanocalver/Documents/Projects/DotCache/configs/prompt_packs/qwen35_cuda_passkey_pack_v1.json)
- [`scripts/run_qwen35_cuda_passkey_probe.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_passkey_probe.py)
- [`scripts/run_qwen35_cuda_passkey_pack_protocol.sh`](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_passkey_pack_protocol.sh)
- [`scripts/summarize_qwen35_cuda_passkey_pack.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/summarize_qwen35_cuda_passkey_pack.py)

### 2. Tier abstraction has cracks

Paper fix:

- rename the concept from "sensitivity tiers" to "policy bundles"
- show K and V candidate order explicitly
- state that bundle labels are implementation shorthands, not a universal conservatism ladder

Code anchor:

- [`planner.py`](/Users/deanocalver/Documents/Projects/DotCache/dotcache/planner.py)

### 3. `exact` is ambiguous

Paper fix:

- reserve `M3 escape` for explicit high-precision storage
- reserve "exact baseline" for no-approximation runtime runs
- use `exact` only as the bundle label meaning "do not adaptively search alternate candidates"

### 4. Sub-linear decode claim is under-argued

Paper fix:

- change the claim to: "selected-set attention can scale sub-linearly while the current selector pass remains linear in candidate pages"
- cite the replay microbench instead of implying the selector is free

Existing evidence:

- [`qwen35_selector_replay_microbench_smoke.json`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_selector_replay_microbench_smoke.json)

Still needed:

- end-to-end selector cost inside the serving loop
- selector + attention + value-mix breakdown at multiple contexts

### 5. Too many codec modes for the amount of validation

Paper fix:

- keep all six modes in the format table because they exist in the page envelope
- explicitly say the current empirical story is mainly about `M0`, `M1`, `M2`, and `M3`
- describe `M4` and `T3` as supported but still lightly validated

Potential future split:

- if `M4` and `T3` stay thinly evaluated, move them to an extension section in the next draft

### 6. Page-level adaptivity is weaker than layer-level heuristics

Paper fix:

- say this directly
- frame Qwen2.5 selective exact-K as the strongest current page-level or layer-head rescue result
- avoid claiming that per-page observed statistics already beat simpler per-layer policies

Still needed:

- fixed per-layer policy vs adaptive per-page routing ablation
- fixed shortlist vs page-stat-conditioned shortlist ablation

### 7. Manual profile discovery is a reproducibility risk

Paper fix:

- call profiles "hand-authored current bundles" instead of "validated final policies"
- move automated discovery from a vague future-work line into a concrete next-step requirement

Existing starting point:

- [`scripts/distill_qwen35_serving_shortlist_rule.py`](/Users/deanocalver/Documents/Projects/DotCache/scripts/distill_qwen35_serving_shortlist_rule.py)
- [`offline_distillation_layer23_mps_cuda_v2.md`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_shortlist_20260329/offline_distillation_layer23_mps_cuda_v2.md)

Still needed:

- calibration/test split for profile search
- frozen-policy held-out evaluation

## High-Value Next Experiments

### A. Grouped-batched decode activation on Qwen3.5 CUDA

Goal:

- understand the small remaining grouped/default performance gap on the Qwen3.5 CUDA lane now that grouped decode is fully operational

Why this is first:

- it is now the clearest systems bottleneck in the shortlist story
- the newer forced-grouped reruns show grouped batching can now run end-to-end on the shortlist rows
- the quality-tail and repro follow-ups now show that grouped mode is repeatable and quality-stable, but still not the default choice

Needed outputs:

- selector / score / mix timing split inside the serving loop
- timing split comparisons for default vs grouped on the same shortlist rows
- one clean rerun table after timing instrumentation, without wrapper interruptions

### B. One stronger `49152` quality rescue

Goal:

- find one quality intervention that is meaningfully stronger than `top_k=8`, without turning the shortlist back into near-exact attention

Negative results already in hand:

- `layer:23:min_ctx` widening does not materially fix the tail
- `top_k=8` modestly helps quality but slows serving and still is not clean
- once `top_k=8` is global, the layer-23 override no longer helps
- grouped decode also does not make `49152` quality-clean

Candidate directions:

- selective value escape only on shortlist pages
- shortlist refine stage on the top few pages
- query-conditioned rescue rule instead of a fixed wider `top_k`

### C. Page-level adaptivity ablation

Goal:

- compare:
  - fixed per-layer bundle
  - fixed per-layer plus explicit rescue overrides
  - current per-page observed-stat routing

Minimal models:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `HuggingFaceTB/SmolLM2-360M-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

### D. Calibration/Test profile search

Goal:

- stop hand-tuning until the plots look nice

Minimum viable version:

- pick a fixed calibration prompt set per model
- search bundle assignments or rescue rules on calibration only
- freeze the policy
- evaluate once on held-out prompts

## Mainline Results Pulled From Other Machines

Latest `main` added fresh ROCm 890M StateCache artifacts and docs:

- [`docs/performance_journal.md`](/Users/deanocalver/Documents/Projects/DotCache/docs/performance_journal.md)
- [`benchmarks/results/qwen35_rocm_890m_statecache_discovery_20260331/`](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_rocm_890m_statecache_discovery_20260331)

These do not directly strengthen the page-selection paper, but they are now merged locally and available if we later broaden the paper's systems comparison section.

## Suggested Order

1. Freeze the manuscript wording cleanup already completed in [`dotcache_layer_page_selection.md`](/Users/deanocalver/Documents/Projects/DotCache/dotcache_layer_page_selection.md).
2. Run one stronger `49152` quality rescue experiment that is not just another wider shortlist.
3. Add timing splits on the grouped CUDA shortlist lane.
4. Run one explicit page-routing ablation against fixed per-layer policies.
5. Only after that, decide whether the paper is still a workshop/system-note submission or ready to target a stronger venue.
