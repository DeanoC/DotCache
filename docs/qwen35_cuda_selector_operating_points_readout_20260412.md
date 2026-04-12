# Qwen3.5-9B CUDA Selector Operating Points Readout

This note captures the current understanding from the partial CUDA operating-point sweep published in:

- [cuda_selector_operating_points_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/cuda_selector_operating_points_summary.md)
- [selector_exploration_report.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/frontier/selector_exploration_report.md)

The goal is to separate what is already established from what is still provisional.

## Executive Read

The CUDA results do not currently show a compelling new selector frontier. What they do show, fairly clearly, is:

1. The promoted weighted selector remains the only fully completed operating point on the stronger `lb21_16_smoke` pack.
2. On the completed weighted point, `quality` and `systems` are tied with `exact` on official score while running much faster.
3. The stronger LongBench pack is exposing a larger issue than selector choice: `exact` itself diverges from `dense` frequently, so selector training/evaluation on that benchmark cannot yet be interpreted as a pure page-selection story.

This means the next priority is not "invent a better selector family." The next priority is:

- finish the missing CUDA operating points on the stronger pack
- audit `dense` vs `exact` parity on CUDA LongBench
- only then decide whether the paper should present this as a selector frontier or as a systems-lane parity result

## What Is Actually Completed

Distinct operating points from the offline frontier:

- `op_weighted_dense_control`
- `op_floor_070`
- `op_floor_075`
- `op_floor_cluster_095_090_085_080`

Completion status on the live packs:

- `task_compare`: complete for all distinct points
- `longbench_mini`: complete for all distinct points
- `longbench_lb21_16_smoke_20260406`:
  - complete for `op_weighted_dense_control`
  - missing for `op_floor_070`
  - missing for `op_floor_075`
  - partial for `op_floor_cluster_095_090_085_080`

So the current recommendation in the published summary is necessarily provisional. It is the best completed point, not the best validated point across the full frontier.

## Strong Signals

### 1. Compact tasks are selector-stable on CUDA

On [cuda_selector_operating_points_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/cuda_selector_operating_points_summary.md), all completed operating points are tied on `task_compare`:

- score `0.667`
- `dense_match_rate = 1.000`
- `accuracy_when_dense_correct = 1.000`

This is good news. It says compact-task behavior on CUDA is not currently being damaged by selector calibration.

### 2. The weighted selector is very fast on CUDA LongBench

On the stronger pack in [op_weighted_dense_control/longbench_selector_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/op_weighted_dense_control/longbench_lb21_16_smoke_20260406/longbench_selector_compare.md):

- `4096 exact`: official `0.323`, `826.3 ms`
- `4096 systems`: official `0.323`, `103.0 ms`
- `8192 exact`: official `0.341`, `1273.7 ms`
- `8192 systems`: official `0.341`, `132.9 ms`

So on the completed weighted point, `systems` preserves the official score while running about `8x` to `10x` faster than `exact`.

### 3. The current "recommended default" is a systems result, not yet a frontier result

The published summary recommends `op_weighted_dense_control`, but that recommendation is driven by completion and stability, not because the sweep proved a better correctness/memory/speed compromise than the unfinished points.

That is fine operationally, but it matters for the paper narrative.

## The Most Important Caveat

### CUDA LongBench is not yet a clean selector benchmark

The strongest current warning is in the same weighted `lb21_16_smoke` report:

- `mean_matches_dense_output` is only about `0.238` at `4096`
- `mean_matches_dense_output` is only about `0.262` at `8192`

That is true for `exact`, `quality`, and `systems`.

This means that on the stronger CUDA LongBench pack, the current DotCache serving path is frequently not reproducing the plain `dense` output exactly, even before we ask whether the selector changed anything. In other words:

- many mismatches are not selector regressions
- the benchmark is partly measuring systems-path parity, not just page-selection quality

This is the single most important interpretation point in the current results.

## Why The LongBench Mini-Pack Is Not Enough

The mini-pack is useful, but it is too small and too forgiving to decide the selector story on CUDA.

For example:

- weighted `longbench_mini` is extremely fast and tied on official score
- more aggressive floor points can become dramatically slower without buying a clear quality win

That is visible in:

- [op_weighted_dense_control/longbench_mini/longbench_selector_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/op_weighted_dense_control/longbench_mini/longbench_selector_compare.md)
- [op_floor_cluster_095_090_085_080/longbench_mini/longbench_selector_compare.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/op_floor_cluster_095_090_085_080/longbench_mini/longbench_selector_compare.md)

The surprising behavior there is that lower-byte floor points can be much slower than weighted on CUDA. So bytes and speed are clearly not aligned automatically.

## What The Offline Frontier Still Tells Us

The offline selector lab in [selector_exploration_report.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_9b_cuda_selector_operating_points_20260407/frontier/selector_exploration_report.md) shows:

- weighted dense-control is the only `1.000 / 1.000` safe/target point
- floor-calibrated points reduce predicted bytes
- but they give up a lot of held-out correctness

That is useful, but offline frontier quality is not enough to justify a paper claim by itself, because the stronger live pack is still incomplete and the dense-parity issue is not yet resolved.

## Current Best Understanding

The current CUDA evidence supports the following claims:

1. DotCache's weighted serving lane can preserve official benchmark score on Qwen3.5-9B LongBench while running much faster than `exact`.
2. Compact-task correctness on CUDA looks stable across the completed selector operating points.
3. The stronger LongBench pack currently exposes a systems-path parity issue (`dense` vs `exact`) that is at least as important as the selector calibration question.

The current CUDA evidence does not yet support the following stronger claims:

1. that a floor-calibrated selector gives a clean correctness/memory/speed frontier on CUDA
2. that weighted is the true best operating point on the stronger pack
3. that LongBench is already a selector-separating benchmark on CUDA

## Immediate Next Steps

### 1. Finish the stronger-pack sweep

Run the missing `lb21_16_smoke` operating points to completion:

- `op_floor_070`
- `op_floor_075`
- `op_floor_cluster_095_090_085_080`

Until that is done, the CUDA operating-point recommendation should be treated as incomplete.

### 2. Do a focused `dense` vs `exact` parity audit on CUDA

This is now a higher priority than more selector retraining.

Questions to answer:

- Why is `exact` only matching `dense` about one quarter of the time on `lb21_16_smoke`?
- Is this prompt formatting, decode contract, stop criteria, scoring cleanup, or another serving-path artifact?
- Which task families account for most of the divergence?

### 3. Keep compact-task dense-preservation as the trusted selector objective

On compact tasks, dense-preservation still looks like the right training and evaluation contract.

On broader LongBench, we should treat dense-preservation as a diagnostic signal, but not yet assume that selector misses are the dominant source of divergence.

### 4. Paper positioning

If the CUDA sweep had to be written up now, the safest claim is:

- the weighted CUDA serving lane preserves official score on completed LongBench checks while decoding much faster than `exact`

The riskier claim would be:

- the CUDA floor sweep already demonstrates a validated selector operating frontier

That second claim should wait until the stronger-pack sweep is finished and the `dense` vs `exact` parity issue is understood.

## Bottom Line

The partial CUDA results are useful, but they are more of a systems-lane result than a finished selector-frontier result.

Right now the best interpretation is:

- compact tasks say the selector is stable
- weighted CUDA LongBench says DotCache can be much faster without hurting official score
- stronger LongBench also says we still need to debug systems-path parity before using it as the main selector-training proof

That should shape the next phase of work.
