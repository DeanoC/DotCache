from __future__ import annotations

import argparse
import json

import torch

from dotcache.integrations.qwen35 import Qwen35TextHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forced dense-only loss harness for Qwen3.5 hybrid text stacks.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--prefix-length", type=int, default=384)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: Qwen35TextHarness,
    *,
    prompt_unit: str,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < sequence_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:sequence_length]

    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_loss.py requires the optional transformers dependencies")
    if args.prefix_length <= 0 or args.prefix_length >= args.sequence_length:
        raise SystemExit("prefix_length must be in [1, sequence_length)")
    if args.eval_steps <= 0 or args.prefix_length + args.eval_steps > args.sequence_length:
        raise SystemExit("prefix_length + eval_steps must be <= sequence_length")

    harness = Qwen35TextHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=args.prompt_unit,
        sequence_length=args.sequence_length,
    )
    result = harness.evaluate_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=args.prefix_length,
        eval_steps=args.eval_steps,
    )
    result.update(
        {
            "benchmark": "qwen35_loss",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "prompt_unit": args.prompt_unit,
            "text_only": True,
            "dotcache_ready": False,
            "hybrid_family": "qwen3_5",
        }
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
