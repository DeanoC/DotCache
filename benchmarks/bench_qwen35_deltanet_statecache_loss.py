from __future__ import annotations

import argparse
import json

import torch

from dotcache.integrations.qwen35 import (
    Qwen35DeltaNetStateHarness,
    parse_qwen35_deltanet_statecache_mode_overrides,
    transformers_available,
)


def _parse_layer_bit_overrides(values: list[str]) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for value in values:
        layer_text, sep, bits_text = str(value).partition(":")
        if sep != ":":
            raise argparse.ArgumentTypeError(f"layer override must look like <layer>:<bits>, got {value!r}")
        overrides[int(layer_text)] = int(bits_text)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-forced readout-only StateCache loss harness for Qwen3.5 DeltaNet layers.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--prefix-length", type=int, default=384)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--layer-bit-overrides", nargs="*", default=[])
    parser.add_argument(
        "--statecache-scope",
        choices=["recurrent_only", "conv_only", "conv_plus_recurrent"],
        default="recurrent_only",
    )
    parser.add_argument("--conv-bits", type=int, default=None)
    parser.add_argument("--conv-layer-bit-overrides", nargs="*", default=[])
    parser.add_argument("--state-stage", choices=["readout_only_m0", "post_update_m0"], default="readout_only_m0")
    parser.add_argument("--renorm-interval", type=int, default=0)
    parser.add_argument("--recurrent-mode-override", action="append", default=[])
    parser.add_argument("--conv-mode-override", action="append", default=[])
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: Qwen35DeltaNetStateHarness,
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
        raise SystemExit("bench_qwen35_deltanet_statecache_loss.py requires the optional transformers dependencies")
    if args.prefix_length <= 0 or args.prefix_length >= args.sequence_length:
        raise SystemExit("prefix_length must be in [1, sequence_length)")
    if args.eval_steps <= 0 or args.prefix_length + args.eval_steps > args.sequence_length:
        raise SystemExit("prefix_length + eval_steps must be <= sequence_length")
    layer_bit_overrides = _parse_layer_bit_overrides(args.layer_bit_overrides)
    conv_layer_bit_overrides = _parse_layer_bit_overrides(args.conv_layer_bit_overrides)

    harness = Qwen35DeltaNetStateHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=args.prompt_unit,
        sequence_length=args.sequence_length,
    )
    recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.recurrent_mode_override)
    conv_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.conv_mode_override)
    result = harness.evaluate_deltanet_statecache_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=args.prefix_length,
        eval_steps=args.eval_steps,
        group_size=args.group_size,
        bits=args.bits,
        layer_bits_overrides=layer_bit_overrides,
        statecache_scope=args.statecache_scope,
        conv_bits=args.conv_bits,
        conv_layer_bits_overrides=conv_layer_bit_overrides,
        state_stage=args.state_stage,
        renorm_interval=args.renorm_interval,
        recurrent_mode_overrides=recurrent_mode_overrides,
        conv_mode_overrides=conv_mode_overrides,
    )
    result.update(
        {
            "benchmark": "qwen35_deltanet_statecache_loss",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "prompt_unit": args.prompt_unit,
            "text_only": True,
            "dotcache_ready": False,
            "hybrid_family": "qwen3_5",
            "deltanet_statecache_scope": args.statecache_scope,
            "deltanet_statecache_conv_bits": args.conv_bits if args.conv_bits is not None else args.bits,
            "deltanet_statecache_layer_bits": {str(layer_id): bits for layer_id, bits in sorted(layer_bit_overrides.items())},
            "deltanet_statecache_conv_layer_bits": {
                str(layer_id): bits for layer_id, bits in sorted(conv_layer_bit_overrides.items())
            },
            "deltanet_statecache_stage_name": args.state_stage,
            "deltanet_statecache_renorm_interval": args.renorm_interval,
        }
    )
    if recurrent_mode_overrides:
        result["deltanet_statecache_recurrent_mode_overrides"] = {
            str(layer_id): mode for layer_id, mode in sorted(recurrent_mode_overrides.items())
        }
    if conv_mode_overrides:
        result["deltanet_statecache_conv_mode_overrides"] = {
            str(layer_id): mode for layer_id, mode in sorted(conv_mode_overrides.items())
        }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
