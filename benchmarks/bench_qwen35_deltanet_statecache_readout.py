from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import (
    Qwen35DeltaNetStateHarness,
    parse_qwen35_deltanet_statecache_renorm_overrides,
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
    parser = argparse.ArgumentParser(description="Prototype 8-bit readout-only StateCache benchmark for Qwen3.5 DeltaNet layers.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=4)
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
    parser.add_argument("--recurrent-renorm-interval-override", action="append", default=[])
    parser.add_argument("--conv-renorm-interval-override", action="append", default=[])
    parser.add_argument("--recurrent-mode-policy", choices=["890m_m3_outlier_pair_midband_v1"], default=None)
    parser.add_argument("--recurrent-mode-override", action="append", default=[])
    parser.add_argument("--readout-recurrent-policy", choices=["890m_context_banded_v1"], default=None)
    parser.add_argument("--readout-recurrent-mode-policy", choices=["890m_m3_outlier_pair_midband_v1"], default=None)
    parser.add_argument("--conv-mode-override", action="append", default=[])
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=[1, 32])
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: Qwen35DeltaNetStateHarness,
    *,
    prompt_unit: str,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _run_case(
    harness: Qwen35DeltaNetStateHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    group_size: int,
    bits: int,
    layer_bits_overrides: dict[int, int],
    statecache_scope: str,
    conv_bits: int | None,
    conv_layer_bits_overrides: dict[int, int],
    state_stage: str,
    renorm_interval: int,
    recurrent_renorm_interval_overrides: dict[int, int],
    conv_renorm_interval_overrides: dict[int, int],
    recurrent_mode_policy: str | None,
    recurrent_mode_overrides: dict[int, str],
    readout_recurrent_policy: str | None,
    readout_recurrent_mode_policy: str | None,
    conv_mode_overrides: dict[int, str],
    base_record: dict[str, object],
    continue_on_error: bool,
) -> None:
    try:
        record = harness.run_deltanet_statecache_readout(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=max_new_tokens,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            statecache_scope=statecache_scope,
            conv_bits=conv_bits,
            conv_layer_bits_overrides=conv_layer_bits_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            conv_mode_overrides=conv_mode_overrides,
        )
    except Exception as exc:  # pragma: no cover - benchmark failure path
        if not continue_on_error:
            raise
        error_record = dict(base_record)
        error_record.update(
            {
                "benchmark": "qwen35_deltanet_statecache_readout",
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "prompt_length": int(input_ids.shape[1]),
            }
        )
        print(json.dumps(error_record, sort_keys=True), flush=True)
        return
    record.update(base_record)
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_deltanet_statecache_readout.py requires the optional transformers dependencies")

    layer_bit_overrides = _parse_layer_bit_overrides(args.layer_bit_overrides)
    conv_layer_bit_overrides = _parse_layer_bit_overrides(args.conv_layer_bit_overrides)
    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", 0) or 0)

    harness = Qwen35DeltaNetStateHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )

    common_record = {
        "benchmark": "qwen35_deltanet_statecache_readout",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "weight_quantization": args.weight_quantization,
        "prompt_unit": args.prompt_unit,
        "model_max_position_embeddings": max_position_embeddings,
        "text_only": True,
        "dotcache_ready": False,
        "hybrid_family": "qwen3_5",
        "deltanet_statecache_group_size": args.group_size,
        "deltanet_statecache_scope": args.statecache_scope,
        "deltanet_statecache_bits": args.bits,
        "deltanet_statecache_conv_bits": args.conv_bits if args.conv_bits is not None else args.bits,
        "deltanet_statecache_layer_bits": {str(layer_id): bits for layer_id, bits in sorted(layer_bit_overrides.items())},
        "deltanet_statecache_conv_layer_bits": {
            str(layer_id): bits for layer_id, bits in sorted(conv_layer_bit_overrides.items())
        },
        "deltanet_statecache_stage_name": args.state_stage,
        "deltanet_statecache_renorm_interval": args.renorm_interval,
    }
    recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.recurrent_mode_override)
    conv_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.conv_mode_override)
    recurrent_renorm_interval_overrides = parse_qwen35_deltanet_statecache_renorm_overrides(
        args.recurrent_renorm_interval_override
    )
    conv_renorm_interval_overrides = parse_qwen35_deltanet_statecache_renorm_overrides(
        args.conv_renorm_interval_override
    )
    if recurrent_mode_overrides:
        common_record["deltanet_statecache_recurrent_mode_overrides"] = {
            str(layer_id): mode for layer_id, mode in sorted(recurrent_mode_overrides.items())
        }
    if conv_mode_overrides:
        common_record["deltanet_statecache_conv_mode_overrides"] = {
            str(layer_id): mode for layer_id, mode in sorted(conv_mode_overrides.items())
        }
    if recurrent_renorm_interval_overrides:
        common_record["deltanet_statecache_recurrent_renorm_interval_overrides"] = {
            str(layer_id): int(interval) for layer_id, interval in sorted(recurrent_renorm_interval_overrides.items())
        }
    if conv_renorm_interval_overrides:
        common_record["deltanet_statecache_conv_renorm_interval_overrides"] = {
            str(layer_id): int(interval) for layer_id, interval in sorted(conv_renorm_interval_overrides.items())
        }

    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            group_size=args.group_size,
            bits=args.bits,
            layer_bits_overrides=layer_bit_overrides,
            statecache_scope=args.statecache_scope,
            conv_bits=args.conv_bits,
            conv_layer_bits_overrides=conv_layer_bit_overrides,
            state_stage=args.state_stage,
            renorm_interval=args.renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_mode_policy=args.recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=args.readout_recurrent_policy,
            readout_recurrent_mode_policy=args.readout_recurrent_mode_policy,
            conv_mode_overrides=conv_mode_overrides,
            base_record={**common_record, "prompt_mode": "repeat_count", "repeat_count": repeat_count},
            continue_on_error=args.continue_on_error,
        )

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            group_size=args.group_size,
            bits=args.bits,
            layer_bits_overrides=layer_bit_overrides,
            statecache_scope=args.statecache_scope,
            conv_bits=args.conv_bits,
            conv_layer_bits_overrides=conv_layer_bit_overrides,
            state_stage=args.state_stage,
            renorm_interval=args.renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            conv_renorm_interval_overrides=conv_renorm_interval_overrides,
            recurrent_mode_policy=args.recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=args.readout_recurrent_policy,
            readout_recurrent_mode_policy=args.readout_recurrent_mode_policy,
            conv_mode_overrides=conv_mode_overrides,
            base_record={**common_record, "prompt_mode": "exact_length", "prompt_length": prompt_length},
            continue_on_error=args.continue_on_error,
        )


if __name__ == "__main__":
    main()
