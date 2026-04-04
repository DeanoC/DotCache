#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dotcache.integrations import Gemma4TextHarness, gemma4_text_recommended_dotcache_config
from dotcache.integrations.llama import resolve_hf_auth_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a Gemma 4 checkpoint through the text-only CausalLM path and run one greedy generation probe.")
    parser.add_argument("--model-id", default="google/gemma-4-E2B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Write one short sentence about cache locality.")
    parser.add_argument("--run-dotcache", action="store_true")
    parser.add_argument(
        "--dotcache-profile",
        choices=["aggressive", "value_exact", "balanced", "exact"],
        default="balanced",
    )
    parser.add_argument("--tokens-per-page", type=int, default=4)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dotcache-backend", default="auto")
    return parser.parse_args()


def _stringify_device_map(device_map: Any) -> dict[str, str] | None:
    if device_map is None:
        return None
    return {str(key): str(value) for key, value in dict(device_map).items()}


def _success_record(
    *,
    args: argparse.Namespace,
    config,
    model,
    prompt_token_count: int,
    generated_token_count: int,
    elapsed_s: float,
    text: str,
) -> dict[str, Any]:
    text_config = getattr(config, "text_config", config)
    layer_types = tuple(getattr(text_config, "layer_types", ()))
    model_device = str(next(model.parameters()).device)
    return {
        "status": "ok",
        "model_id": args.model_id,
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "model_type": str(getattr(config, "model_type", "")),
        "text_model_type": str(getattr(text_config, "model_type", "")),
        "torch_dtype": args.torch_dtype,
        "model_device": model_device,
        "device_map": _stringify_device_map(getattr(model, "hf_device_map", None)),
        "num_hidden_layers": int(getattr(text_config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(text_config, "num_attention_heads", 0)),
        "num_key_value_heads": int(getattr(text_config, "num_key_value_heads", 0)),
        "num_kv_shared_layers": int(getattr(text_config, "num_kv_shared_layers", 0) or 0),
        "sliding_window": int(getattr(text_config, "sliding_window", 0) or 0),
        "max_position_embeddings": int(getattr(text_config, "max_position_embeddings", 0) or 0),
        "full_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "full_attention"),
        "sliding_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "sliding_attention"),
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "elapsed_s": elapsed_s,
        "generated_text": text,
    }


def _dotcache_record(*, args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "dotcache",
        "model_id": args.model_id,
        "dotcache_profile": args.dotcache_profile,
        "tokens_per_page": int(args.tokens_per_page),
        "bits_k": int(args.bits_k),
        "bits_v": int(args.bits_v),
        "group_size": int(args.group_size),
        "dense_generated_ids": list(result["dense_generated_ids"]),
        "dotcache_generated_ids": list(result["dotcache_generated_ids"]),
        "greedy_token_agreement_rate": float(result["greedy_token_agreement_rate"]),
        "teacher_forced_logit_max_abs_error": float(result["teacher_forced_logit_max_abs_error"]),
        "teacher_forced_logit_max_rel_error": float(result["teacher_forced_logit_max_rel_error"]),
        "resident_bytes": int(result["resident_bytes"]),
        "kv_resident_bytes": int(result["kv_resident_bytes"]),
        "decode_ms_per_step": float(result["decode_ms_per_step"]),
        "m0_pages": int(result.get("m0_pages", 0)),
        "m3_pages": int(result.get("m3_pages", 0)),
        "dense_text": result.get("dense_text"),
        "dotcache_text": result.get("dotcache_text"),
    }


def _error_record(args: argparse.Namespace, exc: Exception, *, elapsed_s: float) -> dict[str, Any]:
    return {
        "status": "error",
        "model_id": args.model_id,
        "torch_dtype": args.torch_dtype,
        "elapsed_s": elapsed_s,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()

    try:
        if args.run_dotcache:
            harness = Gemma4TextHarness.from_pretrained(
                args.model_id,
                gemma4_text_recommended_dotcache_config(
                    AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs()),
                    bits_k=args.bits_k,
                    bits_v=args.bits_v,
                    tokens_per_page=args.tokens_per_page,
                    group_size=args.group_size,
                    profile=args.dotcache_profile,
                ),
                backend=args.dotcache_backend,
                device="cuda" if args.device_map == "auto" else args.device_map,
                torch_dtype=args.torch_dtype,
            )
            result = harness.generate_greedy(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
            print(json.dumps(_dotcache_record(args=args, result=result), sort_keys=True))
        else:
            dtype = getattr(torch, args.torch_dtype)
            auth_kwargs = resolve_hf_auth_kwargs()
            config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **auth_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False, **auth_kwargs)
            load_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "device_map": args.device_map,
                "trust_remote_code": False,
                "low_cpu_mem_usage": True,
                **auth_kwargs,
            }
            if args.attn_implementation is not None:
                load_kwargs["attn_implementation"] = args.attn_implementation
            model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
            model.eval()

            inputs = tokenizer(args.prompt, return_tensors="pt")
            inputs = inputs.to(model.device)
            prompt_token_count = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generated_token_count = int(output_ids.shape[-1] - prompt_token_count)
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            elapsed_s = time.perf_counter() - started_at
            print(
                json.dumps(
                    _success_record(
                        args=args,
                        config=config,
                        model=model,
                        prompt_token_count=prompt_token_count,
                        generated_token_count=generated_token_count,
                        elapsed_s=elapsed_s,
                        text=text,
                    ),
                    sort_keys=True,
                )
            )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started_at
        print(json.dumps(_error_record(args, exc, elapsed_s=elapsed_s), sort_keys=True))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
