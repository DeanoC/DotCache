#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotcache.integrations import (
    inspect_nemotron_h_native_config_compatibility,
    load_nemotron_h_remote_config,
    nemotron_h_block_summary,
    nemotron_h_environment_summary,
)
from dotcache.integrations.llama import resolve_hf_auth_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Nemotron-H compatibility on the current machine. "
            "This inspects the published hybrid block layout, checks native AutoConfig compatibility, "
            "and can optionally attempt a dense model load."
        )
    )
    parser.add_argument("--model-id", default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Write one short sentence about caches.")
    parser.add_argument("--attempt-load", action="store_true")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def _write_record(record: dict[str, Any], *, output_path: str | None) -> None:
    if output_path:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


def _build_base_record(*, args: argparse.Namespace, elapsed_s: float) -> dict[str, Any]:
    config = load_nemotron_h_remote_config(args.model_id)
    return {
        "status": "ok",
        "model_id": args.model_id,
        "torch_dtype": args.torch_dtype,
        "requested_device_map": str(args.device_map),
        "elapsed_s": elapsed_s,
        "remote_config_class": type(config).__name__,
        **nemotron_h_environment_summary(),
        **inspect_nemotron_h_native_config_compatibility(args.model_id),
        **nemotron_h_block_summary(config),
        "known_platform_issues": [
            "native_transformers_autoconfig_rejects_published_hybrid_pattern_on_current_5_5_0",
            "remote_code_dense_load_requires_mamba_ssm",
            "rocm_local_mamba_ssm_build_failed_in_shared_python_3_14_env",
        ],
    }


def _attempt_dense_load(args: argparse.Namespace) -> dict[str, Any]:
    auth_kwargs = resolve_hf_auth_kwargs()
    dtype = getattr(torch, args.torch_dtype)
    started_at = time.perf_counter()
    try:
        config = load_nemotron_h_remote_config(args.model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, **auth_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **auth_kwargs,
        )
        model.eval()
        messages = [
            {"role": "system", "content": "Answer directly. Do not show reasoning."},
            {"role": "user", "content": args.prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_token_count = int(inputs.shape[-1])
        return {
            "dense_load_ok": True,
            "dense_load_elapsed_s": float(time.perf_counter() - started_at),
            "model_class": type(model).__name__,
            "model_device": str(next(model.parameters()).device),
            "prompt_token_count": prompt_token_count,
            "generated_token_count": int(output_ids.shape[-1] - prompt_token_count),
            "generated_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "remote_config_class": type(config).__name__,
        }
    except Exception as exc:
        return {
            "dense_load_ok": False,
            "dense_load_elapsed_s": float(time.perf_counter() - started_at),
            "dense_load_error_type": type(exc).__name__,
            "dense_load_error_message": str(exc),
        }


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    record = _build_base_record(args=args, elapsed_s=0.0)
    record["elapsed_s"] = float(time.perf_counter() - started_at)
    if args.attempt_load:
        record.update(_attempt_dense_load(args))
    _write_record(record, output_path=args.output_path)


if __name__ == "__main__":
    main()
