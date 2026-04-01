# Selector Split Suites

These JSON files define ordered suites of frozen selector train/test splits for
`scripts/materialize_page_selector_split_suite.py`.

Each file has the shape:

```json
{
  "suite_name": "example_suite",
  "splits": [
    {
      "split_name": "family_reasoning_holdout",
      "output_subdir": "family_reasoning",
      "holdout_prompt_families": ["reasoning"],
      "holdout_prompt_variants": [],
      "holdout_layers": [],
      "annotations": {"tier": "family"}
    }
  ]
}
```

Recommended workflow:

```bash
./.venv/bin/python scripts/materialize_page_selector_split_suite.py \
  --input-dir /path/to/oracle_bundle \
  --output-root /path/to/frozen_splits \
  --suite-config configs/selector_split_suites/local_smoke_suite.json

./.venv/bin/python scripts/train_page_selector_split_batch.py \
  --split-manifest /path/to/frozen_splits/split_manifest.json \
  --output-dir /path/to/batch_eval
```

Single-command local smoke path:

```bash
bash scripts/run_page_selector_local_smoke_suite.sh /path/to/oracle_bundle
```

Single-command larger-machine path:

```bash
bash scripts/run_page_selector_larger_machine_suite.sh /path/to/output_root
```

If you want the first stronger-model lane pinned to Qwen3.5 4B, use:

```bash
bash scripts/run_page_selector_qwen35_4b_suite.sh /path/to/output_root
```

That wrapper runs capture, oracle labeling, suite materialization, and batch
selector evaluation using `larger_machine_comprehensive_suite.json`, with a
stable output layout under `capture/`, `labels/`, `suite/`, and `batch_eval/`.

Included templates:

- `local_smoke_suite.json`
  Small, fast sanity suite for local MPS and tiny CUDA runs.
- `prompt_generalization_suite.json`
  Family and variant holdouts that stress style generalization.
- `larger_machine_comprehensive_suite.json`
  Broader suite for larger GPU boxes, including family, variant, and
  family-plus-layer slices.
