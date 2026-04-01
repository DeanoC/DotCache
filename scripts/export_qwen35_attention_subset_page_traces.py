#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from dotcache.integrations.qwen35 import Qwen35AttentionSubsetHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture tiny Qwen3.5 attention-subset page traces for local oracle replay.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", action="append", choices=["K", "V"], default=[])
    return parser.parse_args()


def _build_exact_length_inputs(harness: Qwen35AttentionSubsetHarness, *, prompt_unit: str, prompt_length: int):
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
        raise SystemExit("export_qwen35_attention_subset_page_traces.py requires the optional transformers dependencies")

    harness = Qwen35AttentionSubsetHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    kinds = tuple(args.kind) if args.kind else ("K", "V")
    if args.prompt is not None:
        result = harness.capture_attention_subset_page_traces(
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
        result = harness.capture_attention_subset_page_traces(
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
