from __future__ import annotations

import argparse
import gc
import json
from typing import Any

import torch

from dotcache.integrations.qwen35 import (
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


def _parse_case(value: str) -> tuple[int, int]:
    prefix_text, sep, eval_text = str(value).partition(":")
    if sep != ":":
        raise argparse.ArgumentTypeError(f"case must look like <prefix_length>:<eval_steps>, got {value!r}")
    prefix_length = int(prefix_text)
    eval_steps = int(eval_text)
    if prefix_length <= 0 or eval_steps <= 0:
        raise argparse.ArgumentTypeError(f"case values must be positive, got {value!r}")
    return prefix_length, eval_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a StateCache-first Qwen3.5 regression suite over fixed teacher-forced slices."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--cases", nargs="*", default=["64:8", "128:16", "256:16"])
    parser.add_argument(
        "--statecache-scopes",
        nargs="*",
        choices=["recurrent_only", "conv_only", "conv_plus_recurrent"],
        default=["recurrent_only", "conv_only", "conv_plus_recurrent"],
    )
    parser.add_argument(
        "--localization-scopes",
        nargs="*",
        choices=["recurrent_only", "conv_only", "conv_plus_recurrent"],
        default=["recurrent_only", "conv_plus_recurrent"],
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--layer-bit-overrides", nargs="*", default=[])
    parser.add_argument("--conv-bits", type=int, default=None)
    parser.add_argument("--conv-layer-bit-overrides", nargs="*", default=[])
    parser.add_argument("--state-stage", choices=["readout_only_m0", "post_update_m0"], default="post_update_m0")
    parser.add_argument("--renorm-interval", type=int, default=0)
    parser.add_argument("--recurrent-mode-override", action="append", default=[])
    parser.add_argument("--conv-mode-override", action="append", default=[])
    parser.add_argument("--output-format", choices=["jsonl", "json"], default="jsonl")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def _build_exact_length_inputs(
    *,
    tokenizer: Any,
    device: torch.device,
    prompt_unit: str,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")
    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < sequence_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:sequence_length]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _clear_accelerator_cache(device: str | None) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _dense_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dense_teacher_forced_loss": float(result["dense_teacher_forced_loss"]),
        "dense_teacher_forced_perplexity": float(result["dense_teacher_forced_perplexity"]),
        "dense_teacher_forced_target_match_rate": float(result["dense_teacher_forced_target_match_rate"]),
        "dense_decode_ms_per_step": float(result["dense_decode_ms_per_step"]),
    }


def _statecache_loss_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": str(result["deltanet_statecache_scope"]),
        "teacher_forced_loss": float(result["deltanet_statecache_teacher_forced_loss"]),
        "teacher_forced_perplexity": float(result["deltanet_statecache_teacher_forced_perplexity"]),
        "teacher_forced_target_match_rate": float(result["deltanet_statecache_teacher_forced_target_match_rate"]),
        "teacher_forced_loss_delta": float(result["teacher_forced_loss_delta"]),
        "teacher_forced_perplexity_ratio": float(result["teacher_forced_perplexity_ratio"]),
        "decode_ms_per_step": float(result["deltanet_statecache_decode_ms_per_step"]),
        "dense_conv_state_bytes": int(result["deltanet_conv_state_bytes"]),
        "dense_recurrent_state_bytes": int(result["deltanet_recurrent_state_bytes"]),
        "statecache_conv_state_bytes": int(result["deltanet_statecache_conv_state_bytes"]),
        "statecache_recurrent_state_bytes": int(result["deltanet_statecache_recurrent_state_bytes"]),
        "statecache_fixed_resident_bytes": int(result["deltanet_statecache_fixed_resident_bytes"]),
        "effective_conv_compression_ratio": float(result["deltanet_statecache_effective_conv_compression_ratio"]),
        "effective_recurrent_compression_ratio": float(result["deltanet_statecache_effective_recurrent_compression_ratio"]),
        "effective_fixed_resident_compression_ratio": float(
            result["deltanet_statecache_effective_fixed_resident_compression_ratio"]
        ),
        "recurrent_mode_overrides": dict(result.get("deltanet_statecache_recurrent_mode_overrides", {})),
        "conv_mode_overrides": dict(result.get("deltanet_statecache_conv_mode_overrides", {})),
    }


def _statecache_localization_summary(result: dict[str, Any]) -> dict[str, Any]:
    per_step = [float(value) for value in result.get("deltanet_statecache_per_step_logit_max_abs_error", [])]
    return {
        "scope": str(result["deltanet_statecache_scope"]),
        "first_divergence_step": result.get("deltanet_statecache_first_divergence_step"),
        "first_failure_layer": result.get("deltanet_statecache_first_failure_layer"),
        "first_recurrent_failure_layer": result.get("deltanet_statecache_first_recurrent_failure_layer"),
        "first_conv_failure_layer": result.get("deltanet_statecache_first_conv_failure_layer"),
        "first_combined_failure_layer": result.get("deltanet_statecache_first_combined_failure_layer"),
        "max_per_step_logit_abs_error": max(per_step, default=0.0),
        "per_step_logit_max_abs_error": per_step,
    }


def _run_dense_case(
    harness: Qwen35TextHarness,
    *,
    prompt_unit: str,
    prefix_length: int,
    eval_steps: int,
) -> dict[str, Any]:
    sequence_length = int(prefix_length + eval_steps)
    input_ids, attention_mask = _build_exact_length_inputs(
        tokenizer=harness.tokenizer,
        device=harness.adapter.device,
        prompt_unit=prompt_unit,
        sequence_length=sequence_length,
    )
    return harness.evaluate_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=prefix_length,
        eval_steps=eval_steps,
    )


def _run_statecache_case(
    harness: Qwen35DeltaNetStateHarness,
    *,
    prompt_unit: str,
    prefix_length: int,
    eval_steps: int,
    group_size: int,
    bits: int,
    layer_bits_overrides: dict[int, int],
    statecache_scope: str,
    conv_bits: int | None,
    conv_layer_bits_overrides: dict[int, int],
    state_stage: str,
    renorm_interval: int,
    recurrent_mode_overrides: dict[int, str],
    conv_mode_overrides: dict[int, str],
    with_localization: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sequence_length = int(prefix_length + eval_steps)
    input_ids, attention_mask = _build_exact_length_inputs(
        tokenizer=harness.tokenizer,
        device=harness.adapter.device,
        prompt_unit=prompt_unit,
        sequence_length=sequence_length,
    )
    loss_result = harness.evaluate_deltanet_statecache_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=prefix_length,
        eval_steps=eval_steps,
        group_size=group_size,
        bits=bits,
        layer_bits_overrides=layer_bits_overrides,
        statecache_scope=statecache_scope,
        conv_bits=conv_bits,
        conv_layer_bits_overrides=conv_layer_bits_overrides,
        state_stage=state_stage,
        renorm_interval=renorm_interval,
        recurrent_mode_overrides=recurrent_mode_overrides,
        conv_mode_overrides=conv_mode_overrides,
    )
    localization_result = None
    if with_localization:
        localization_result = run_qwen35_deltanet_statecache_localization_harness(
            harness.model,
            harness.adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=harness.tokenizer,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            statecache_scope=statecache_scope,
            conv_bits=conv_bits,
            conv_layer_bits_overrides=conv_layer_bits_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_mode_overrides=recurrent_mode_overrides,
            conv_mode_overrides=conv_mode_overrides,
        )
    return loss_result, localization_result


def _record_error(
    *,
    case_id: str,
    scope: str,
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "scope": scope,
        "stage": stage,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_statecache_regression_suite.py requires the optional transformers dependencies")

    cases = [_parse_case(value) for value in args.cases]
    localization_scopes = set(args.localization_scopes)
    invalid_localization = localization_scopes.difference(args.statecache_scopes)
    if invalid_localization:
        raise SystemExit(f"localization scopes must be a subset of statecache scopes, got {sorted(invalid_localization)}")

    layer_bits_overrides = _parse_layer_bit_overrides(args.layer_bit_overrides)
    conv_layer_bits_overrides = _parse_layer_bit_overrides(args.conv_layer_bit_overrides)
    recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.recurrent_mode_override)
    conv_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.conv_mode_override)

    records: list[dict[str, Any]] = []

    text_harness = Qwen35TextHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    dense_results_by_case: dict[str, dict[str, Any]] = {}
    for prefix_length, eval_steps in cases:
        case_id = f"{prefix_length}:{eval_steps}"
        dense_results_by_case[case_id] = _run_dense_case(
            text_harness,
            prompt_unit=args.prompt_unit,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
        )
    del text_harness
    _clear_accelerator_cache(args.device)

    statecache_harness = Qwen35DeltaNetStateHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    for prefix_length, eval_steps in cases:
        case_id = f"{prefix_length}:{eval_steps}"
        record: dict[str, Any] = {
            "benchmark": "qwen35_statecache_regression_suite",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "prompt_unit": args.prompt_unit,
            "case_id": case_id,
            "sequence_length": int(prefix_length + eval_steps),
            "prefix_length": int(prefix_length),
            "eval_steps": int(eval_steps),
            "state_stage": args.state_stage,
            "statecache_bits": int(args.bits),
            "statecache_conv_bits": int(args.conv_bits if args.conv_bits is not None else args.bits),
            "statecache_scopes": list(args.statecache_scopes),
            "localization_scopes": list(args.localization_scopes),
            "dense": _dense_summary(dense_results_by_case[case_id]),
            "statecache": {},
        }
        for scope in args.statecache_scopes:
            try:
                loss_result, localization_result = _run_statecache_case(
                    statecache_harness,
                    prompt_unit=args.prompt_unit,
                    prefix_length=prefix_length,
                    eval_steps=eval_steps,
                    group_size=args.group_size,
                    bits=args.bits,
                    layer_bits_overrides=layer_bits_overrides,
                    statecache_scope=scope,
                    conv_bits=args.conv_bits,
                    conv_layer_bits_overrides=conv_layer_bits_overrides,
                    state_stage=args.state_stage,
                    renorm_interval=args.renorm_interval,
                    recurrent_mode_overrides=recurrent_mode_overrides,
                    conv_mode_overrides=conv_mode_overrides,
                    with_localization=scope in localization_scopes,
                )
            except Exception as exc:  # pragma: no cover - benchmark failure path
                if not args.continue_on_error:
                    raise
                record["statecache"][scope] = _record_error(case_id=case_id, scope=scope, stage="statecache", exc=exc)
                continue
            mode_summary: dict[str, Any] = {
                "status": "ok",
                "loss": _statecache_loss_summary(loss_result),
            }
            if localization_result is not None:
                mode_summary["localization"] = _statecache_localization_summary(localization_result)
            record["statecache"][scope] = mode_summary
        records.append(record)

    if args.output_format == "json":
        print(json.dumps({"records": records}, sort_keys=True), flush=True)
        return

    for record in records:
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
