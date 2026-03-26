# DotCache NVIDIA bootstrap guide for non-MPS Llama development

This guide is for the current Linux + NVIDIA path in this repo.

Today that path means:

- Hugging Face Llama-family models run on `cuda`
- DotCache decode stays on `cpu_ref`
- the existing Llama integration harness is used to validate correctness and workflow before a CUDA DotCache backend exists

## Scope

Use this machine setup when you want to work on:

- the non-MPS Llama integration path
- CUDA-hosted dense prefill and decode comparisons
- model-loading, prompt-shaping, and replay/generation harness work
- future CUDA backend development from an already-working NVIDIA environment

Do not treat this as a claim that DotCache already has a CUDA execution backend. It does not.

## Bootstrap

From the repo root:

```bash
bash scripts/bootstrap_nvidia_llama_dev.sh
```

That script:

- creates `.venv`
- upgrades `pip`, `setuptools`, and `wheel`
- installs `torch==2.4.1` by default
- installs the repo with `.[dev,hf]`
- verifies `torch.cuda.is_available()`

If you update the NVIDIA driver later and want to try a newer PyTorch build, override the install target explicitly:

```bash
TORCH_SPEC='torch>=2.5' bash scripts/bootstrap_nvidia_llama_dev.sh
```

## Verification

Check the environment directly:

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Run the tiny random Llama smoke benchmark on CUDA:

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --random-tiny --backend cpu_ref --device cuda --max-new-tokens 4
```

Run the Llama integration test file:

```bash
.venv/bin/python -m pytest -q tests/test_llama_integration.py
```

The MPS-specific test in that file is skipped automatically on NVIDIA systems.

## Working rule on NVIDIA

Use `--device cuda` when you want the model on the GPU.

Use `--backend cpu_ref` when you want DotCache decode to remain on the implemented reference path.

That combination is the intended non-MPS development baseline for this repo until a CUDA backend is added.

## First real-model command

```bash
.venv/bin/python benchmarks/bench_llama_decode.py --backend cpu_ref --device cuda --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Write one short sentence about cache locality." --max-new-tokens 8
```
