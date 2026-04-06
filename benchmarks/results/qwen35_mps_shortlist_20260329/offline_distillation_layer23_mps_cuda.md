# Layer 23 Offline Distillation Report

- examples: `20`
- bad recall threshold: `0.4`
- bad examples: `2`

## Top Rules

### Rule 1

- clauses: `approx_boundary_margin_normalized >= 0.0558`
- precision: `0.667`
- recall: `1.000`
- f1: `0.800`
- confusion: `tp=2 fp=1 fn=0`
- predicted groups:
  - `step=0 kv=1 recall=0.700 anchor=4 recent_old=4 run=4 margin=0.055803044166097486`
  - `step=1 kv=1 recall=0.364 anchor=4 recent_old=4 run=4 margin=0.08467993893329272`
  - `step=0 kv=1 recall=0.364 anchor=0 recent_old=4 run=4 margin=0.16742035813401482`

### Rule 2

- clauses: `anchor_pages >= 0; approx_boundary_margin_normalized >= 0.0558`
- precision: `0.667`
- recall: `1.000`
- f1: `0.800`
- confusion: `tp=2 fp=1 fn=0`
- predicted groups:
  - `step=0 kv=1 recall=0.700 anchor=4 recent_old=4 run=4 margin=0.055803044166097486`
  - `step=1 kv=1 recall=0.364 anchor=4 recent_old=4 run=4 margin=0.08467993893329272`
  - `step=0 kv=1 recall=0.364 anchor=0 recent_old=4 run=4 margin=0.16742035813401482`

### Rule 3

- clauses: `recent_old_pages >= 0; approx_boundary_margin_normalized >= 0.0558`
- precision: `0.667`
- recall: `1.000`
- f1: `0.800`
- confusion: `tp=2 fp=1 fn=0`
- predicted groups:
  - `step=0 kv=1 recall=0.700 anchor=4 recent_old=4 run=4 margin=0.055803044166097486`
  - `step=1 kv=1 recall=0.364 anchor=4 recent_old=4 run=4 margin=0.08467993893329272`
  - `step=0 kv=1 recall=0.364 anchor=0 recent_old=4 run=4 margin=0.16742035813401482`

### Rule 4

- clauses: `recent_old_pages >= 4; approx_boundary_margin_normalized >= 0.0558`
- precision: `0.667`
- recall: `1.000`
- f1: `0.800`
- confusion: `tp=2 fp=1 fn=0`
- predicted groups:
  - `step=0 kv=1 recall=0.700 anchor=4 recent_old=4 run=4 margin=0.055803044166097486`
  - `step=1 kv=1 recall=0.364 anchor=4 recent_old=4 run=4 margin=0.08467993893329272`
  - `step=0 kv=1 recall=0.364 anchor=0 recent_old=4 run=4 margin=0.16742035813401482`

### Rule 5

- clauses: `recent_old_run_length >= 0; approx_boundary_margin_normalized >= 0.0558`
- precision: `0.667`
- recall: `1.000`
- f1: `0.800`
- confusion: `tp=2 fp=1 fn=0`
- predicted groups:
  - `step=0 kv=1 recall=0.700 anchor=4 recent_old=4 run=4 margin=0.055803044166097486`
  - `step=1 kv=1 recall=0.364 anchor=4 recent_old=4 run=4 margin=0.08467993893329272`
  - `step=0 kv=1 recall=0.364 anchor=0 recent_old=4 run=4 margin=0.16742035813401482`

