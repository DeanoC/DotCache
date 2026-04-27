# DotCache Paper Benchmark Runner

This repo is currently focused on regenerating the paper benchmarks for
DotCache with the corrected certified attention implementation.

The benchmark entrypoint is:

```bash
benchmarks/paper/run_distributed_quality_slice.py
```

It is designed for running independent paper-quality slices across one or more
GPU machines.

## Hardware Target

Validated target:

- NVIDIA RTX PRO 6000, 96 GB VRAM
- CUDA-visible PyTorch
- Python 3.11+
- `NousResearch/Meta-Llama-3.1-8B`
- INT8 model weights via bitsandbytes

64K paper-quality slices currently take about 95-98 minutes each on the RTX PRO
6000. PG-19 dominates the runtime.

## Fresh Machine Setup

```bash
git clone git@github.com:DeanoC/DotCache.git
cd DotCache
git checkout port-to-paper-20260424

export HF_TOKEN=...
source scripts/env_cuda.sh
sudo apt-get update
sudo apt-get install -y \
  libcusparse-dev-13-1 \
  libcublas-dev-13-1 \
  libcusolver-dev-13-1
bash scripts/bootstrap_nvidia_llama_dev.sh
```

The bootstrap script creates `.venv`, installs the repo with the required dev
and Hugging Face dependencies, and fails if CUDA is not visible inside the
virtualenv.

The CUDA dev packages above provide the headers required to build the native
Blackwell extension:

- `libcusparse-dev-13-1`: `cusparse.h`
- `libcublas-dev-13-1`: `cublas_v2.h`
- `libcusolver-dev-13-1`: `cusolverDn.h`

Check the environment:

```bash
.venv/bin/python - <<'PY'
import torch
import transformers
import bitsandbytes

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Run the focused tests:

```bash
.venv/bin/python -m pytest \
  tests/test_hybrid_int4_attention.py \
  tests/test_adaptive_topk.py \
  tests/test_experiment_v2_sweep_runner.py \
  -q
```

Expected on the current branch: `36 passed`.

## Slice Semantics

One `--slice-id` maps to one deterministic unit of each benchmark:

- PG-19: held-out chunk index.
- NIAH: paired trial index, valid `0..99`.
- RULER: deterministic sample index.

For PG-19 confidence intervals, each context needs 20 chunks:

```text
slices 0..19 = full 20-chunk PG-19 CI for one context
```

The runner default is `--cache-mode full-bounded`. This is the intended quality
run mode: it uses a bounded GPU scratch/cache large enough to avoid page-in
noise while still recording the bounded cache size in the manifest. Do not use
`full-mirror` for paper quality runs.

## Run One Slice

Dry-run first:

```bash
.venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 7 \
  --mode context \
  --context 65536 \
  --output-dir runs/paper_v2_distributed_64k_machineA \
  --dry-run
```

Run the slice:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 7 \
  --mode context \
  --context 65536 \
  --output-dir runs/paper_v2_distributed_64k_machineA \
  --resume
```

This runs PG-19, NIAH, and RULER for context 64K and slice 7.

To run only PG-19:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 7 \
  --mode context \
  --context 65536 \
  --benches pg19 \
  --output-dir runs/paper_v2_distributed_64k_machineA \
  --resume
```

## Run Slice Ranges Across Machines

Assign disjoint slice ranges. Example: if slices `0..6` are already complete,
the remaining 64K PG-19 CI slices are `7..19`.

Machine A:

```bash
for sid in 7 8 9 10 11 12; do
  PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
    --slice-id "$sid" \
    --mode context \
    --context 65536 \
    --output-dir runs/paper_v2_distributed_64k_machineA \
    --resume
done
```

Machine B:

```bash
for sid in 13 14 15 16 17 18 19; do
  PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
    --slice-id "$sid" \
    --mode context \
    --context 65536 \
    --output-dir runs/paper_v2_distributed_64k_machineB \
    --resume
done
```

On comparable GPUs, this split should complete in about 10-12 hours, limited by
the seven-slice machine.

## Contexts

Paper contexts:

```text
8192
32768
65536
131072
```

The runner also supports running one slice across multiple contexts:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 0 \
  --mode line \
  --contexts 8192 32768 65536 \
  --output-dir runs/paper_v2_distributed_line0 \
  --resume
```

Use 128K selectively; it is much more expensive.

## Monitoring

Each benchmark writes a JSON output and a log. Example:

```bash
tail -f runs/paper_v2_distributed_64k_machineA/slice_0007/64K/pg19_slice_0007_64K.log
cat runs/paper_v2_distributed_64k_machineA/slice_0007/manifest.json
```

The manifest records:

- host
- git commit and branch
- dirty worktree status
- exact commands
- cache mode and cache block counts
- output paths
- exit codes
- wall times

## Expected 64K Timings

Observed on RTX PRO 6000 96 GB:

- PG-19 one chunk: about 89-92 minutes.
- NIAH one paired trial: about 1 minute.
- RULER one sample across seven subtasks: about 5-6 minutes.
- Full context slice: about 95-98 minutes.

## Result Checks

Before using results in the paper:

- Every assigned slice must have `exit_code: 0` in `manifest.json`.
- PG-19 dense/certified perplexity ratios should be close to 1.0.
- NIAH dense-pass/certified-fail critical failures should be zero.
- RULER critical failures should be reviewed and aggregated before conclusions.

Quick parse example:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("runs/paper_v2_distributed_64k_machineA")
for manifest in sorted(root.glob("slice_*/manifest.json")):
    m = json.loads(manifest.read_text())
    sid = m["slice_id"]
    ok = all(j["exit_code"] == 0 for j in m.get("completed", []))
    print(f"slice {sid:04d}: ok={ok}")
PY
```

## Collecting Results

Copy completed `runs/paper_v2_distributed_*` directories back to the main
machine. Keep output directory names unique per machine to avoid overwriting
manifests.

Do not edit JSON outputs by hand. The paper table generation should consume the
raw JSONs directly.

## Pushing Result Archives

After a machine completes its assigned slices, create and push a compressed
archive of that machine's output directory. Result JSON/logs compress heavily,
so this is cheap and gives us a stable artifact for later auditing.

Example for machine A:

```bash
mkdir -p runs/archives
tar -czf runs/archives/paper_v2_64k_machineA_slices_7_12.tar.gz \
  runs/paper_v2_distributed_64k_machineA

git add runs/archives/paper_v2_64k_machineA_slices_7_12.tar.gz \
  runs/paper_v2_distributed_64k_machineA
git commit -m "Add 64K paper slices 7-12"
git push origin port-to-paper-20260424
```

If the loose result directory is too noisy for a given machine, at minimum push
the `.tar.gz` archive and the slice summary. Do not include old calibration,
profile, cache, or debug artifacts unless they are explicitly needed.

## More Detail

The longer benchmark runbook is:

```text
docs/paper_v2_distributed_runbook.md
```
