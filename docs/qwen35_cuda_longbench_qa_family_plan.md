# Qwen3.5 LongBench Closure Plan

This note replaces the earlier QA mini-pack plan as the main paper-facing benchmark target.

The old mini, medium, and full QA packs still matter, but only as cheap regression gates. They are no longer the benchmark courtroom for the paper.

## Current Milestone

- one combined benchmark-closure milestone
- courtroom model: `Qwen/Qwen3.5-9B`
- benchmark coverage target: full original LongBench suite
- external parity rows:
  - `streaming_sink_recent`
  - `quest_like`
- fairness views:
  - `matched_quality`
  - `matched_memory`

## Implementation Path On Branch

- task registry: [longbench_v1.py](/workspace/DotCache/dotcache/longbench_v1.py)
- benchmark harness: [bench_qwen35_attention_subset_dotcache_longbench_qa.py](/workspace/DotCache/benchmarks/bench_qwen35_attention_subset_dotcache_longbench_qa.py)
- compare runner: [run_qwen35_longbench_selector_compare.py](/workspace/DotCache/scripts/run_qwen35_longbench_selector_compare.py)
- pack runner: [run_qwen35_longbench_pack.py](/workspace/DotCache/scripts/run_qwen35_longbench_pack.py)
- report: [report_qwen35_longbench_selector_compare.py](/workspace/DotCache/scripts/report_qwen35_longbench_selector_compare.py)
- failure workbook: [report_qwen35_longbench_failure_workbook.py](/workspace/DotCache/scripts/report_qwen35_longbench_failure_workbook.py)

## Default Entry Points

Main 9B shell wrappers now target the closure path by default:

```bash
scripts/run_qwen35_9b_longbench_selector_compare.sh
scripts/run_qwen35_9b_longbench_pack.sh <output-dir>
```

Those wrappers now:

- enumerate the original LongBench suite from the official `data.zip`
- run `exact`, `quality`, `systems`, `streaming_sink_recent`, and `quest_like`
- emit both the main comparison report and the failure workbook

## Expected Outputs

For a standard run directory, the branch now expects:

- `qwen35_9b_longbench_selector_compare.jsonl`
- `longbench_selector_compare.md`
- `longbench_selector_compare.json`
- `longbench_failure_workbook.md`
- `longbench_failure_workbook.json`

## Reporting Contract

The main comparison report must expose:

- official LongBench task score
- task-family breakdown
- worst-dataset floor
- decode ms/step and `p95`
- effective bytes per token
- matched-quality parity picks
- matched-memory parity picks

The failure workbook must record every held-out `systems` miss versus `exact` and classify it as:

- `selection_miss`
- `write_format_damage`
- `downstream_under_attention`

## Historical Note

The earlier LongBench QA mini-pack branch work is still useful for:

- fast smoke checks
- wrapper regressions
- targeted Hotpot-style diagnostics

It should not be cited as the main external quality closure result once the full-suite `Qwen3.5-9B` run exists.
