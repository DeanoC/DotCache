from __future__ import annotations

import argparse
import json
from dataclasses import replace

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe per-layer K/V sensitivity tiers on one loaded Llama-family model."
    )
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--prefix-length", type=int, default=384)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--probe-kind", choices=["K", "V", "both"], default="both")
    parser.add_argument("--probe-tier", choices=["strict", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: LlamaDotCacheHarness,
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


def _select_probe_layers(num_hidden_layers: int, *, explicit_layers: list[int], max_layers: int) -> list[int]:
    if explicit_layers:
        layers = sorted(set(int(layer) for layer in explicit_layers if 0 <= int(layer) < num_hidden_layers))
        if not layers:
            raise ValueError("no valid explicit layers were provided")
        return layers
    if num_hidden_layers <= 0:
        return []
    capped = max(1, min(int(max_layers), num_hidden_layers))
    if capped >= num_hidden_layers:
        return list(range(num_hidden_layers))
    if capped == 1:
        return [num_hidden_layers - 1]
    step = (num_hidden_layers - 1) / float(capped - 1)
    selected = sorted({int(round(index * step)) for index in range(capped)})
    while len(selected) < capped:
        for layer in range(num_hidden_layers):
            if layer not in selected:
                selected.append(layer)
            if len(selected) == capped:
                break
    return sorted(selected)


def _build_config(args: argparse.Namespace) -> DotCacheConfig:
    model_config = AutoConfig.from_pretrained(args.model_id)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    return DotCacheConfig(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        default_mode_k="M0",
        default_mode_v="M0",
        key_policy_tier="exact",
        value_policy_tier="exact",
        tokens_per_page=args.tokens_per_page,
    )


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_layer_sensitivity.py requires the optional transformers dependencies")
    if args.prefix_length <= 0 or args.prefix_length >= args.sequence_length:
        raise SystemExit("prefix_length must be in [1, sequence_length)")
    if args.eval_steps <= 0 or args.prefix_length + args.eval_steps > args.sequence_length:
        raise SystemExit("prefix_length + eval_steps must be <= sequence_length")

    base_config = _build_config(args)
    harness = LlamaDotCacheHarness.from_pretrained(
        args.model_id,
        base_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=args.prompt_unit,
        sequence_length=args.sequence_length,
    )

    num_hidden_layers = int(harness.model.config.num_hidden_layers)
    probe_layers = _select_probe_layers(
        num_hidden_layers,
        explicit_layers=list(args.layers),
        max_layers=args.max_layers,
    )

    baseline_result: dict[str, object] | None = None
    if args.include_baseline:
        harness.adapter.reconfigure(base_config, backend=args.backend)
        baseline = harness.evaluate_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_length=args.prefix_length,
            eval_steps=args.eval_steps,
        )
        baseline_result = {
            **baseline,
            "benchmark": "layer_sensitivity",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "sequence_length": args.sequence_length,
            "prefix_length": args.prefix_length,
            "eval_steps": args.eval_steps,
            "probe_kind": "baseline",
            "probe_tier": "exact",
            "probe_layer": -1,
        }
        print(json.dumps(baseline_result, sort_keys=True), flush=True)

    baseline_loss = float(baseline_result["teacher_forced_loss_delta"]) if baseline_result is not None else 0.0
    del baseline_loss  # explicit: exact baseline loss delta is always 0 relative to itself

    probe_kinds = ["K", "V"] if args.probe_kind == "both" else [args.probe_kind]
    base_dotcache_loss = None
    if baseline_result is not None:
        base_dotcache_loss = float(baseline_result["dotcache_teacher_forced_loss"])

    for kind in probe_kinds:
        for layer in probe_layers:
            if kind == "K":
                config = replace(
                    base_config,
                    key_policy_tier="exact",
                    value_policy_tier="exact",
                    key_layer_sensitivity=(f"layer:{layer}={args.probe_tier}",),
                    value_layer_sensitivity=(),
                )
            else:
                config = replace(
                    base_config,
                    key_policy_tier="exact",
                    value_policy_tier="exact",
                    key_layer_sensitivity=(),
                    value_layer_sensitivity=(f"layer:{layer}={args.probe_tier}",),
                )
            harness.adapter.reconfigure(config, backend=args.backend)
            result = harness.evaluate_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prefix_length=args.prefix_length,
                eval_steps=args.eval_steps,
            )
            enriched = {
                **result,
                "benchmark": "layer_sensitivity",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "sequence_length": args.sequence_length,
                "prefix_length": args.prefix_length,
                "eval_steps": args.eval_steps,
                "probe_kind": kind,
                "probe_tier": args.probe_tier,
                "probe_layer": int(layer),
                "key_layer_sensitivity": list(config.key_layer_sensitivity),
                "value_layer_sensitivity": list(config.value_layer_sensitivity),
            }
            if base_dotcache_loss is not None:
                enriched["probe_vs_exact_dotcache_loss_delta"] = float(
                    float(result["dotcache_teacher_forced_loss"]) - base_dotcache_loss
                )
            print(json.dumps(enriched, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
