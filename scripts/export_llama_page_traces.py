#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, resolve_hf_auth_kwargs, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Llama-family page traces for oracle replay.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", action="append", choices=["K", "V"], default=[])
    return parser.parse_args()


def _build_dotcache_config(*, model_id: str, tokens_per_page: int, group_size: int) -> DotCacheConfig:
    auth_kwargs = resolve_hf_auth_kwargs()
    model_config = AutoConfig.from_pretrained(model_id, **auth_kwargs)
    head_dim = int(model_config.hidden_size) // int(model_config.num_attention_heads)
    return DotCacheConfig(
        head_dim=head_dim,
        group_size=int(group_size),
        bits_k=4,
        bits_v=4,
        tokens_per_page=int(tokens_per_page),
    )


def _build_exact_length_inputs(harness: LlamaDotCacheHarness, *, prompt_unit: str, prompt_length: int):
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")

    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < prompt_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:prompt_length]

    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def main() -> int:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("export_llama_page_traces.py requires the optional transformers dependencies")

    harness = LlamaDotCacheHarness.from_pretrained(
        args.model_id,
        _build_dotcache_config(
            model_id=args.model_id,
            tokens_per_page=args.tokens_per_page,
            group_size=args.group_size,
        ),
        backend="cpu_ref",
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    kinds = tuple(args.kind) if args.kind else ("K", "V")
    if args.prompt is not None:
        result = harness.capture_page_traces(
            prompt=args.prompt,
            decode_steps=args.decode_steps,
            output_dir=args.output_dir,
            tokens_per_page=args.tokens_per_page,
            kinds=kinds,
        )
    else:
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=args.prompt_length,
        )
        result = harness.capture_page_traces(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=args.decode_steps,
            output_dir=args.output_dir,
            tokens_per_page=args.tokens_per_page,
            kinds=kinds,
        )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
