# Paper v2 Distributed Benchmark Runbook

This is the setup and execution path for the clean paper rerun on NVIDIA CUDA
machines. It is the current Llama paper benchmark path, not the older Qwen
selector documentation.

## Machine Requirements

- NVIDIA GPU with enough VRAM for the requested context. The 64K quality slices
  have been validated on an RTX PRO 6000 96 GB.
- CUDA-visible PyTorch environment.
- Python 3.11 or newer.
- Enough local disk for the Hugging Face model/cache and benchmark outputs.
- Hugging Face access for `NousResearch/Meta-Llama-3.1-8B`.

## Fresh Machine Setup

```bash
git clone git@github.com:DeanoC/DotCache.git
cd DotCache
git checkout port-to-paper-20260424

export HF_TOKEN=...
source scripts/env_cuda.sh
bash scripts/bootstrap_nvidia_llama_dev.sh
```

The bootstrap script creates `.venv`, installs the repo with the `dev` and `hf`
extras, and fails fast if CUDA is not visible inside the virtualenv.

Run a quick environment check:

```bash
.venv/bin/python - <<'PY'
import torch
import transformers
import bitsandbytes
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("bitsandbytes", bitsandbytes.__version__)
PY
```

Run the focused correctness tests:

```bash
.venv/bin/python -m pytest \
  tests/test_hybrid_int4_attention.py \
  tests/test_adaptive_topk.py \
  tests/test_experiment_v2_sweep_runner.py \
  -q
```

Expected result on the current branch is `36 passed`.

## Distributed Slice Semantics

Use `benchmarks/paper/run_distributed_quality_slice.py`.

One `slice-id` means:

- PG-19: one held-out chunk index.
- NIAH: one paired trial index, valid `0..99`.
- RULER: one deterministic sample index.

For PG-19 20-chunk confidence intervals, each context needs slices `0..19`.
For the 64K run currently in progress, slices `0..6` have been assigned on the
first machine, so the remaining 64K PG-19 CI slices are `7..19`.

The default cache mode is `full-bounded`, which uses an effectively unbounded
bounded GPU cache for quality runs while still labelling the scratch/cache size
in the manifest. Do not use `full-mirror` for paper quality runs.

## Single Slice Commands

Dry-run first:

```bash
.venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 7 \
  --mode context \
  --context 65536 \
  --output-dir runs/paper_v2_distributed_64k_machineA \
  --dry-run
```

Run one 64K context slice:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/paper/run_distributed_quality_slice.py \
  --slice-id 7 \
  --mode context \
  --context 65536 \
  --output-dir runs/paper_v2_distributed_64k_machineA \
  --resume
```

This runs PG-19, NIAH, and RULER for that slice/context. On the RTX PRO 6000,
observed 64K time is about 95-98 minutes per full context slice.

## Multi-Machine Slice Ranges

Assign disjoint slice ranges. Example for finishing the remaining 64K PG-19
20-chunk CI after slices `0..6`:

Machine A, six slices:

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

Machine B, seven slices:

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

On comparable GPUs this should complete in roughly 10-12 hours, with the
seven-slice machine as the wall-clock limiter.

If you only want PG-19 on those machines, add `--benches pg19`. That saves only
about 6 minutes per 64K slice because PG-19 dominates runtime.

## Monitoring

Each slice writes a manifest and per-benchmark logs:

```bash
tail -f runs/paper_v2_distributed_64k_machineA/slice_0007/64K/pg19_slice_0007_64K.log
cat runs/paper_v2_distributed_64k_machineA/slice_0007/manifest.json
```

The manifest records host, git commit, branch, dirty status, commands, cache
mode, outputs, exit codes, and wall times.

## Result Collection

Copy the completed `runs/paper_v2_distributed_*` directories back to the main
machine. Keep the directory names unique per machine to avoid overwriting
manifests.

Before using results in the paper, check:

- Every assigned slice has `exit_code: 0` in `manifest.json`.
- PG-19 dense/certified perplexity ratios are near 1.0.
- NIAH critical failures are zero.
- RULER critical failures are reviewed individually; one-sample slices can show
  task-level noise, so aggregate before drawing conclusions.

## Known Current Timings

Observed on RTX PRO 6000 96 GB at 64K:

- PG-19 one chunk: about 89-92 minutes.
- NIAH one paired trial: about 1 minute.
- RULER one sample across seven subtasks: about 5-6 minutes.
- Full context slice: about 95-98 minutes.

