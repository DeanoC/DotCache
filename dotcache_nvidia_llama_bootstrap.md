# DotCache NVIDIA bootstrap guide for 5090-era CUDA development

This guide is for the current Linux + NVIDIA path in this repo.

Today that path means:

- Hugging Face Llama-family and Qwen2-family models run on `cuda`
- DotCache exact decode can run on the implemented `torch_cuda` backend
- the RTX 5090 pod is the preferred box for larger-model HF scale-up work
- the optional vLLM path is still a follow-on milestone, not the first bring-up target

## Scope

Use this machine setup when you want to work on:

- exact HF model integration on CUDA
- CUDA-hosted dense prefill and DotCache decode comparisons
- larger-model benchmark recording on the 5090 pod
- future vLLM bring-up from an already-working NVIDIA environment

## Bootstrap

From the repo root:

```bash
bash scripts/bootstrap_nvidia_llama_dev.sh
```

That script:

- creates `.venv`
- upgrades `pip`, `setuptools`, and `wheel`
- reuses an already-working system CUDA torch when present
- otherwise installs `torch>=2.8` by default
- installs the repo with `.[dev,hf]`
- verifies `torch.cuda.is_available()`

If you want to override the fallback install target explicitly:

```bash
TORCH_SPEC='torch>=2.8' bash scripts/bootstrap_nvidia_llama_dev.sh
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

Run the CUDA backend and integration test files:

```bash
.venv/bin/python -m pytest -q tests/test_torch_cuda_backend.py
.venv/bin/python -m pytest -q tests/test_llama_integration.py
.venv/bin/python -m pytest -q tests/test_qwen2_integration.py
```

The MPS-specific test in that file is skipped automatically on NVIDIA systems.

## Working rule on NVIDIA

Use `--device cuda` when you want the model on the GPU.

Use `--backend torch_cuda` when you want the implemented CUDA DotCache path.

Use `--backend cpu_ref` only when you explicitly want the reference-oracle path for comparison.

## First scale-up commands

```bash
bash scripts/run_qwen25_compare_cuda.sh
bash scripts/run_qwen25_7b_compare_cuda.sh
bash scripts/run_llama32_compare_cuda.sh
```

To summarize recorded runs afterward:

```bash
.venv/bin/python scripts/report_model_benchmarks.py --benchmark qwen2_compare
.venv/bin/python scripts/report_model_benchmarks.py --benchmark llama_compare
```
