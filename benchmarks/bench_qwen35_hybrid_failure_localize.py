from __future__ import annotations

import argparse
import json

import torch

from dotcache.config import DotCacheConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheHarness,
    Qwen35DeltaNetStateHarness,
    Qwen35TextHarness,
    parse_qwen35_deltanet_statecache_mode_overrides,
    run_qwen35_deltanet_statecache_localization_harness,
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


def _build_exact_length_inputs(harness: Qwen35TextHarness | Qwen35AttentionSubsetDotCacheHarness | Qwen35DeltaNetStateHarness, *, prompt_unit: str, prompt_length: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize Qwen3.5 hybrid drift across dense, DotCache-only, StateCache-only, and combined modes.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--sequence-length", type=int, default=160)
    parser.add_argument("--prefix-length", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--statecache-bits", type=int, default=8)
    parser.add_argument("--statecache-layer-bit-overrides", nargs="*", default=[])
    parser.add_argument(
        "--statecache-scope",
        choices=["recurrent_only", "conv_only", "conv_plus_recurrent"],
        default="recurrent_only",
    )
    parser.add_argument("--statecache-conv-bits", type=int, default=None)
    parser.add_argument("--statecache-conv-layer-bit-overrides", nargs="*", default=[])
    parser.add_argument("--statecache-stage", choices=["readout_only_m0", "post_update_m0"], default="post_update_m0")
    parser.add_argument("--statecache-renorm-interval", type=int, default=0)
    parser.add_argument("--statecache-recurrent-mode-override", action="append", default=[])
    parser.add_argument("--statecache-conv-mode-override", action="append", default=[])
    return parser.parse_args()


def _first_layer_from_map(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    for layer_key in sorted(value, key=lambda item: int(item)):
        if float(value[layer_key]) > 1e-6:
            return int(layer_key)
    return None


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_hybrid_failure_localize.py requires the optional transformers dependencies")

    layer_bit_overrides = _parse_layer_bit_overrides(args.statecache_layer_bit_overrides)
    conv_layer_bit_overrides = _parse_layer_bit_overrides(args.statecache_conv_layer_bit_overrides)
    recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.statecache_recurrent_mode_override)
    conv_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.statecache_conv_mode_override)

    dotcache_config = DotCacheConfig(
        head_dim=256,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        tokens_per_page=args.tokens_per_page,
    )

    text_harness = Qwen35TextHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    dotcache_harness = Qwen35AttentionSubsetDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config=dotcache_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    statecache_harness = Qwen35DeltaNetStateHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    input_ids, attention_mask = _build_exact_length_inputs(
        text_harness,
        prompt_unit=args.prompt_unit,
        prompt_length=args.sequence_length,
    )
    prefix_input_ids = input_ids[:, : args.prefix_length]
    prefix_attention_mask = attention_mask[:, : args.prefix_length]

    dense_result = text_harness.evaluate_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=args.prefix_length,
        eval_steps=args.eval_steps,
    )
    dotcache_result = dotcache_harness.run_attention_subset_dotcache(
        input_ids=prefix_input_ids,
        attention_mask=prefix_attention_mask,
        decode_steps=args.eval_steps - 1,
    )
    statecache_result = run_qwen35_deltanet_statecache_localization_harness(
        statecache_harness.model,
        statecache_harness.adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=statecache_harness.tokenizer,
        prefix_length=args.prefix_length,
        eval_steps=args.eval_steps,
        group_size=args.group_size,
        bits=args.statecache_bits,
        layer_bits_overrides=layer_bit_overrides,
        statecache_scope=args.statecache_scope,
        conv_bits=args.statecache_conv_bits,
        conv_layer_bits_overrides=conv_layer_bit_overrides,
        state_stage=args.statecache_stage,
        renorm_interval=args.statecache_renorm_interval,
        recurrent_mode_overrides=recurrent_mode_overrides,
        conv_mode_overrides=conv_mode_overrides,
    )
    combined_result = dotcache_harness.run_hybrid_combined_localization(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=args.prefix_length,
        eval_steps=args.eval_steps,
        statecache_group_size=args.group_size,
        statecache_bits=args.statecache_bits,
        statecache_layer_bits_overrides=layer_bit_overrides,
        statecache_scope=args.statecache_scope,
        statecache_conv_bits=args.statecache_conv_bits,
        statecache_conv_layer_bits_overrides=conv_layer_bit_overrides,
        statecache_stage=args.statecache_stage,
        statecache_renorm_interval=args.statecache_renorm_interval,
        statecache_recurrent_mode_overrides=recurrent_mode_overrides,
        statecache_conv_mode_overrides=conv_mode_overrides,
    )

    summary = {
        "benchmark": "qwen35_hybrid_failure_localize",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "sequence_length": args.sequence_length,
        "prefix_length": args.prefix_length,
        "eval_steps": args.eval_steps,
        "tokens_per_page": args.tokens_per_page,
        "statecache_scope": args.statecache_scope,
        "statecache_bits": args.statecache_bits,
        "statecache_conv_bits": args.statecache_conv_bits if args.statecache_conv_bits is not None else args.statecache_bits,
        "statecache_stage": args.statecache_stage,
        "statecache_renorm_interval": args.statecache_renorm_interval,
        "statecache_layer_bit_overrides": {str(layer_id): bits for layer_id, bits in sorted(layer_bit_overrides.items())},
        "statecache_conv_layer_bit_overrides": {
            str(layer_id): bits for layer_id, bits in sorted(conv_layer_bit_overrides.items())
        },
        "statecache_recurrent_mode_overrides": {str(layer_id): mode for layer_id, mode in sorted(recurrent_mode_overrides.items())},
        "statecache_conv_mode_overrides": {str(layer_id): mode for layer_id, mode in sorted(conv_mode_overrides.items())},
        "dense_teacher_forced_loss": dense_result["dense_teacher_forced_loss"],
        "dotcache_first_attention_failure_layer": _first_layer_from_map(
            dotcache_result.get("replay_output_max_abs_error_by_layer")
        ),
        "dotcache_attention_output_max_abs_error_by_layer": dotcache_result.get("replay_output_max_abs_error_by_layer", {}),
        "dotcache_teacher_forced_logit_max_abs_error": dotcache_result.get("teacher_forced_logit_max_abs_error", 0.0),
        "statecache_first_divergence_step": statecache_result.get("deltanet_statecache_first_divergence_step"),
        "statecache_first_failure_layer": statecache_result.get("deltanet_statecache_first_failure_layer"),
        "statecache_first_recurrent_failure_layer": statecache_result.get("deltanet_statecache_first_recurrent_failure_layer"),
        "statecache_first_conv_failure_layer": statecache_result.get("deltanet_statecache_first_conv_failure_layer"),
        "statecache_first_combined_failure_layer": statecache_result.get("deltanet_statecache_first_combined_failure_layer"),
        "statecache_per_step_logit_max_abs_error": statecache_result.get("deltanet_statecache_per_step_logit_max_abs_error", []),
        "combined_first_divergence_step": combined_result.get("combined_first_divergence_step"),
        "combined_first_attention_failure_layer": combined_result.get("combined_first_attention_failure_layer"),
        "combined_first_recurrent_failure_layer": combined_result.get("combined_first_recurrent_failure_layer"),
        "combined_first_conv_failure_layer": combined_result.get("combined_first_conv_failure_layer"),
        "combined_first_failure_family": combined_result.get("combined_first_failure_family"),
        "combined_per_step_logit_max_abs_error": combined_result.get("combined_per_step_logit_max_abs_error", []),
        "combined_attention_output_max_abs_error_by_layer": combined_result.get("combined_attention_output_max_abs_error_by_layer", {}),
        "dotcache_result": dotcache_result,
        "statecache_result": statecache_result,
        "combined_result": combined_result,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
