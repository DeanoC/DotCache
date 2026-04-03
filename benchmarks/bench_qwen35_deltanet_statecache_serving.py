from __future__ import annotations

import argparse
import json
import statistics

import torch
from transformers import AutoConfig

from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import (
    Qwen35DeltaNetStateHarness,
    parse_qwen35_deltanet_statecache_int_overrides,
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
    parser = argparse.ArgumentParser(description="Serving-style StateCache benchmark for Qwen3.5 DeltaNet layers.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--layer-bit-overrides", nargs="*", default=[])
    parser.add_argument("--recurrent-group-size-policy", choices=["890m_long_horizon_group_escape_v1"], default=None)
    parser.add_argument("--recurrent-layer-group-size-override", action="append", default=[])
    parser.add_argument("--state-stage", choices=["readout_only_m0", "post_update_m0"], default="readout_only_m0")
    parser.add_argument("--renorm-interval", type=int, default=0)
    parser.add_argument("--recurrent-renorm-interval-override", action="append", default=[])
    parser.add_argument("--recurrent-mode-policy", choices=["890m_m3_outlier_pair_midband_v1"], default=None)
    parser.add_argument("--recurrent-mode-override", action="append", default=[])
    parser.add_argument("--paired-recurrent-group-size-policy", choices=["890m_long_horizon_group_escape_v1"], default=None)
    parser.add_argument("--paired-recurrent-layer-group-size-override", action="append", default=[])
    parser.add_argument("--paired-recurrent-mode-policy", choices=["890m_m3_outlier_pair_midband_v1"], default=None)
    parser.add_argument("--paired-recurrent-mode-override", action="append", default=[])
    parser.add_argument("--paired-label", default="candidate")
    parser.add_argument("--paired-order-schedule", choices=["AB", "BA", "ABBA", "BAAB"], default="AB")
    parser.add_argument("--readout-recurrent-policy", choices=["890m_context_banded_v1"], default=None)
    parser.add_argument("--readout-recurrent-mode-policy", choices=["890m_m3_outlier_pair_midband_v1"], default=None)
    parser.add_argument("--readout-recurrent-renorm-interval-override", action="append", default=[])
    parser.add_argument("--readout-recurrent-mode-override", action="append", default=[])
    parser.add_argument("--warmup-in-process-repeats", type=int, default=0)
    parser.add_argument("--in-process-repeats", type=int, default=1)
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


def _collect_case_record(
    harness: Qwen35DeltaNetStateHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    group_size: int,
    bits: int,
    layer_bits_overrides: dict[int, int],
    recurrent_group_size_policy: str | None,
    recurrent_layer_group_size_overrides: dict[int, int],
    state_stage: str,
    renorm_interval: int,
    recurrent_renorm_interval_overrides: dict[int, int],
    recurrent_mode_policy: str | None,
    recurrent_mode_overrides: dict[int, str],
    readout_recurrent_policy: str | None,
    readout_recurrent_mode_policy: str | None,
    readout_recurrent_renorm_interval_overrides: dict[int, int],
    readout_recurrent_mode_overrides: dict[int, str],
    continue_on_error: bool,
) -> dict[str, object]:
    try:
        record = harness.run_deltanet_statecache_serving(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=max_new_tokens,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            recurrent_group_size_policy=recurrent_group_size_policy,
            recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
            readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
        )
    except Exception as exc:  # pragma: no cover - benchmark failure path
        if not continue_on_error:
            raise
        return {
            "benchmark": "qwen35_deltanet_statecache_serving",
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "prompt_length": int(input_ids.shape[1]),
        }
    return record


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _summarize_in_process_repeat_records(
    records: list[dict[str, object]],
    *,
    warmup_in_process_repeats: int,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must be non-empty")
    aggregate = dict(records[-1])
    decode_values = [float(record["deltanet_statecache_decode_ms_per_step"]) for record in records]
    prefill_values = [float(record["prefill_ms"]) for record in records]
    prefill_reserved_values = [
        int(record.get("deltanet_statecache_prefill_cuda_peak_memory_reserved_bytes", 0)) for record in records
    ]
    decode_reserved_values = [
        int(record.get("deltanet_statecache_decode_cuda_peak_memory_reserved_bytes", 0)) for record in records
    ]
    generated_ids = [record.get("deltanet_statecache_generated_ids") for record in records]
    aggregate["benchmark_measurement_mode"] = "in_process_repeated"
    aggregate["warmup_in_process_repeats"] = int(warmup_in_process_repeats)
    aggregate["in_process_repeats"] = int(len(records))
    aggregate["in_process_repeat_decode_ms_per_step_values"] = decode_values
    aggregate["in_process_repeat_prefill_ms_values"] = prefill_values
    aggregate["in_process_repeat_prefill_cuda_peak_memory_reserved_bytes_values"] = prefill_reserved_values
    aggregate["in_process_repeat_decode_cuda_peak_memory_reserved_bytes_values"] = decode_reserved_values
    aggregate["in_process_repeat_generated_ids_consistent"] = all(ids == generated_ids[0] for ids in generated_ids[1:])
    aggregate["in_process_repeat_decode_ms_per_step_stddev"] = _pstdev(decode_values)
    aggregate["in_process_repeat_prefill_ms_stddev"] = _pstdev(prefill_values)
    aggregate["deltanet_statecache_decode_ms_per_step"] = _mean(decode_values)
    aggregate["prefill_ms"] = _mean(prefill_values)
    aggregate["deltanet_statecache_prefill_cuda_peak_memory_reserved_bytes"] = int(round(_mean(prefill_reserved_values)))
    aggregate["deltanet_statecache_decode_cuda_peak_memory_reserved_bytes"] = int(round(_mean(decode_reserved_values)))
    return aggregate


def _summarize_paired_in_process_repeat_records(
    baseline_records: list[dict[str, object]],
    candidate_records: list[dict[str, object]],
    *,
    warmup_in_process_repeats: int,
    in_process_repeats: int,
    candidate_label: str,
    order_schedule: str,
) -> dict[str, object]:
    if not baseline_records or not candidate_records:
        raise ValueError("paired records must be non-empty")
    if len(baseline_records) != len(candidate_records):
        raise ValueError("paired record lists must have the same length")
    baseline_summary = _summarize_in_process_repeat_records(
        baseline_records,
        warmup_in_process_repeats=warmup_in_process_repeats,
    )
    candidate_summary = _summarize_in_process_repeat_records(
        candidate_records,
        warmup_in_process_repeats=warmup_in_process_repeats,
    )
    baseline_decode = float(baseline_summary["deltanet_statecache_decode_ms_per_step"])
    candidate_decode = float(candidate_summary["deltanet_statecache_decode_ms_per_step"])
    baseline_prefill = float(baseline_summary["prefill_ms"])
    candidate_prefill = float(candidate_summary["prefill_ms"])
    paired_generated_ids_match = all(
        baseline_record.get("deltanet_statecache_generated_ids") == candidate_record.get("deltanet_statecache_generated_ids")
        for baseline_record, candidate_record in zip(baseline_records, candidate_records)
    )
    baseline_ratio = float(baseline_summary.get("deltanet_statecache_effective_recurrent_compression_ratio", 1.0))
    candidate_ratio = float(candidate_summary.get("deltanet_statecache_effective_recurrent_compression_ratio", 1.0))
    return {
        "benchmark_measurement_mode": "in_process_paired_repeated",
        "warmup_in_process_repeats": int(warmup_in_process_repeats),
        "in_process_repeats": int(in_process_repeats),
        "paired_order_schedule": str(order_schedule),
        "paired_baseline_sample_count": int(len(baseline_records)),
        "paired_candidate_sample_count": int(len(candidate_records)),
        "paired_candidate_label": str(candidate_label),
        "paired_generated_ids_match_all": paired_generated_ids_match,
        "baseline_decode_ms_per_step": baseline_decode,
        "baseline_decode_ms_per_step_values": baseline_summary["in_process_repeat_decode_ms_per_step_values"],
        "baseline_decode_ms_per_step_stddev": baseline_summary["in_process_repeat_decode_ms_per_step_stddev"],
        "baseline_prefill_ms": baseline_prefill,
        "baseline_prefill_ms_values": baseline_summary["in_process_repeat_prefill_ms_values"],
        "baseline_prefill_ms_stddev": baseline_summary["in_process_repeat_prefill_ms_stddev"],
        "baseline_recurrent_compression_ratio": baseline_ratio,
        "candidate_decode_ms_per_step": candidate_decode,
        "candidate_decode_ms_per_step_values": candidate_summary["in_process_repeat_decode_ms_per_step_values"],
        "candidate_decode_ms_per_step_stddev": candidate_summary["in_process_repeat_decode_ms_per_step_stddev"],
        "candidate_prefill_ms": candidate_prefill,
        "candidate_prefill_ms_values": candidate_summary["in_process_repeat_prefill_ms_values"],
        "candidate_prefill_ms_stddev": candidate_summary["in_process_repeat_prefill_ms_stddev"],
        "candidate_recurrent_compression_ratio": candidate_ratio,
        "paired_decode_ms_per_step_delta": float(candidate_decode - baseline_decode),
        "paired_decode_ms_per_step_ratio": float(candidate_decode / max(baseline_decode, 1e-8)),
        "paired_prefill_ms_delta": float(candidate_prefill - baseline_prefill),
        "paired_prefill_ms_ratio": float(candidate_prefill / max(baseline_prefill, 1e-8)),
        "paired_recurrent_compression_ratio_delta": float(candidate_ratio - baseline_ratio),
    }


def _run_case(
    harness: Qwen35DeltaNetStateHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    group_size: int,
    bits: int,
    layer_bits_overrides: dict[int, int],
    recurrent_group_size_policy: str | None,
    recurrent_layer_group_size_overrides: dict[int, int],
    state_stage: str,
    renorm_interval: int,
    recurrent_renorm_interval_overrides: dict[int, int],
    recurrent_mode_policy: str | None,
    recurrent_mode_overrides: dict[int, str],
    readout_recurrent_policy: str | None,
    readout_recurrent_mode_policy: str | None,
    readout_recurrent_renorm_interval_overrides: dict[int, int],
    readout_recurrent_mode_overrides: dict[int, str],
    warmup_in_process_repeats: int,
    in_process_repeats: int,
    base_record: dict[str, object],
    continue_on_error: bool,
) -> None:
    if in_process_repeats <= 0:
        raise ValueError("in_process_repeats must be positive")
    for _ in range(max(int(warmup_in_process_repeats), 0)):
        warmup_record = _collect_case_record(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            recurrent_group_size_policy=recurrent_group_size_policy,
            recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
            readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
            continue_on_error=continue_on_error,
        )
        if warmup_record.get("status") == "error":
            warmup_record.update(base_record)
            warmup_record["benchmark_measurement_mode"] = "warmup_error"
            warmup_record["warmup_in_process_repeats"] = int(warmup_in_process_repeats)
            warmup_record["in_process_repeats"] = int(in_process_repeats)
            print(json.dumps(warmup_record, sort_keys=True), flush=True)
            return
    records: list[dict[str, object]] = []
    for _ in range(int(in_process_repeats)):
        record = _collect_case_record(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            group_size=group_size,
            bits=bits,
            layer_bits_overrides=layer_bits_overrides,
            recurrent_group_size_policy=recurrent_group_size_policy,
            recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
            state_stage=state_stage,
            renorm_interval=renorm_interval,
            recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
            recurrent_mode_policy=recurrent_mode_policy,
            recurrent_mode_overrides=recurrent_mode_overrides,
            readout_recurrent_policy=readout_recurrent_policy,
            readout_recurrent_mode_policy=readout_recurrent_mode_policy,
            readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
            readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
            continue_on_error=continue_on_error,
        )
        if record.get("status") == "error":
            record.update(base_record)
            record["benchmark_measurement_mode"] = "single_shot_error"
            record["warmup_in_process_repeats"] = int(warmup_in_process_repeats)
            record["in_process_repeats"] = int(in_process_repeats)
            print(json.dumps(record, sort_keys=True), flush=True)
            return
        records.append(record)
    record_to_emit = (
        _summarize_in_process_repeat_records(records, warmup_in_process_repeats=int(warmup_in_process_repeats))
        if int(in_process_repeats) > 1 or int(warmup_in_process_repeats) > 0
        else dict(records[0], benchmark_measurement_mode="single_shot", warmup_in_process_repeats=0, in_process_repeats=1)
    )
    record_to_emit.update(base_record)
    print(json.dumps(record_to_emit, sort_keys=True), flush=True)


def _run_paired_case(
    harness: Qwen35DeltaNetStateHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    group_size: int,
    bits: int,
    layer_bits_overrides: dict[int, int],
    recurrent_group_size_policy: str | None,
    recurrent_layer_group_size_overrides: dict[int, int],
    state_stage: str,
    renorm_interval: int,
    recurrent_renorm_interval_overrides: dict[int, int],
    paired_recurrent_group_size_policy: str | None,
    paired_recurrent_layer_group_size_overrides: dict[int, int],
    baseline_recurrent_mode_policy: str | None,
    baseline_recurrent_mode_overrides: dict[int, str],
    paired_recurrent_mode_policy: str | None,
    paired_recurrent_mode_overrides: dict[int, str],
    readout_recurrent_policy: str | None,
    readout_recurrent_mode_policy: str | None,
    readout_recurrent_renorm_interval_overrides: dict[int, int],
    readout_recurrent_mode_overrides: dict[int, str],
    warmup_in_process_repeats: int,
    in_process_repeats: int,
    paired_label: str,
    paired_order_schedule: str,
    base_record: dict[str, object],
    continue_on_error: bool,
) -> None:
    if in_process_repeats <= 0:
        raise ValueError("in_process_repeats must be positive")
    baseline_kwargs = {
        "harness": harness,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "group_size": group_size,
        "bits": bits,
        "layer_bits_overrides": layer_bits_overrides,
        "recurrent_group_size_policy": recurrent_group_size_policy,
        "recurrent_layer_group_size_overrides": recurrent_layer_group_size_overrides,
        "state_stage": state_stage,
        "renorm_interval": renorm_interval,
        "recurrent_renorm_interval_overrides": recurrent_renorm_interval_overrides,
        "recurrent_mode_policy": baseline_recurrent_mode_policy,
        "recurrent_mode_overrides": baseline_recurrent_mode_overrides,
        "readout_recurrent_policy": readout_recurrent_policy,
        "readout_recurrent_mode_policy": readout_recurrent_mode_policy,
        "readout_recurrent_renorm_interval_overrides": readout_recurrent_renorm_interval_overrides,
        "readout_recurrent_mode_overrides": readout_recurrent_mode_overrides,
        "continue_on_error": continue_on_error,
    }
    candidate_kwargs = dict(baseline_kwargs)
    candidate_kwargs.update(
        {
            "recurrent_group_size_policy": paired_recurrent_group_size_policy,
            "recurrent_layer_group_size_overrides": paired_recurrent_layer_group_size_overrides,
            "recurrent_mode_policy": paired_recurrent_mode_policy,
            "recurrent_mode_overrides": paired_recurrent_mode_overrides,
        }
    )
    paired_order_map = {
        "AB": ("A", "B"),
        "BA": ("B", "A"),
        "ABBA": ("A", "B", "B", "A"),
        "BAAB": ("B", "A", "A", "B"),
    }
    if paired_order_schedule not in paired_order_map:
        raise ValueError(f"unknown paired_order_schedule {paired_order_schedule!r}")
    order_sequence = paired_order_map[paired_order_schedule]
    for _ in range(max(int(warmup_in_process_repeats), 0)):
        for order_code in order_sequence:
            kwargs = baseline_kwargs if order_code == "A" else candidate_kwargs
            warmup_record = _collect_case_record(**kwargs)
            if warmup_record.get("status") == "error":
                warmup_record.update(base_record)
                warmup_record["benchmark_measurement_mode"] = "paired_warmup_error"
                warmup_record["warmup_in_process_repeats"] = int(warmup_in_process_repeats)
                warmup_record["in_process_repeats"] = int(in_process_repeats)
                warmup_record["paired_order_schedule"] = str(paired_order_schedule)
                print(json.dumps(warmup_record, sort_keys=True), flush=True)
                return
    baseline_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    for _ in range(int(in_process_repeats)):
        for order_code in order_sequence:
            kwargs = baseline_kwargs if order_code == "A" else candidate_kwargs
            record = _collect_case_record(**kwargs)
            if record.get("status") == "error":
                record.update(base_record)
                record["benchmark_measurement_mode"] = (
                    "paired_baseline_error" if order_code == "A" else "paired_candidate_error"
                )
                record["warmup_in_process_repeats"] = int(warmup_in_process_repeats)
                record["in_process_repeats"] = int(in_process_repeats)
                record["paired_order_schedule"] = str(paired_order_schedule)
                print(json.dumps(record, sort_keys=True), flush=True)
                return
            if order_code == "A":
                baseline_records.append(record)
            else:
                candidate_records.append(record)
    record_to_emit = _summarize_paired_in_process_repeat_records(
        baseline_records,
        candidate_records,
        warmup_in_process_repeats=int(warmup_in_process_repeats),
        in_process_repeats=int(in_process_repeats),
        candidate_label=str(paired_label),
        order_schedule=str(paired_order_schedule),
    )
    record_to_emit.update(base_record)
    print(json.dumps(record_to_emit, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_deltanet_statecache_serving.py requires the optional transformers dependencies")
    if args.in_process_repeats <= 0:
        raise SystemExit("in_process_repeats must be positive")
    if args.warmup_in_process_repeats < 0:
        raise SystemExit("warmup_in_process_repeats must be non-negative")

    layer_bit_overrides = _parse_layer_bit_overrides(args.layer_bit_overrides)
    recurrent_layer_group_size_overrides = parse_qwen35_deltanet_statecache_int_overrides(
        args.recurrent_layer_group_size_override,
        value_name="group_size",
        minimum=1,
    )
    paired_recurrent_layer_group_size_overrides = parse_qwen35_deltanet_statecache_int_overrides(
        args.paired_recurrent_layer_group_size_override,
        value_name="group_size",
        minimum=1,
    )
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
        "benchmark": "qwen35_deltanet_statecache_serving",
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
        "deltanet_statecache_bits": args.bits,
        "deltanet_statecache_layer_bits": {str(layer_id): bits for layer_id, bits in sorted(layer_bit_overrides.items())},
        "deltanet_statecache_stage_name": args.state_stage,
        "deltanet_statecache_renorm_interval": args.renorm_interval,
    }
    recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.recurrent_mode_override)
    paired_recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(args.paired_recurrent_mode_override)
    readout_recurrent_mode_overrides = parse_qwen35_deltanet_statecache_mode_overrides(
        args.readout_recurrent_mode_override
    )
    readout_recurrent_renorm_interval_overrides = parse_qwen35_deltanet_statecache_renorm_overrides(
        args.readout_recurrent_renorm_interval_override
    )
    recurrent_renorm_interval_overrides = parse_qwen35_deltanet_statecache_renorm_overrides(
        args.recurrent_renorm_interval_override
    )
    if recurrent_mode_overrides:
        common_record["deltanet_statecache_recurrent_mode_overrides"] = {
            str(layer_id): mode for layer_id, mode in sorted(recurrent_mode_overrides.items())
        }
    if recurrent_layer_group_size_overrides:
        common_record["deltanet_statecache_recurrent_layer_group_size_overrides"] = {
            str(layer_id): int(group) for layer_id, group in sorted(recurrent_layer_group_size_overrides.items())
        }
    if args.recurrent_group_size_policy is not None:
        common_record["deltanet_statecache_recurrent_group_size_policy"] = str(args.recurrent_group_size_policy)
    if recurrent_renorm_interval_overrides:
        common_record["deltanet_statecache_recurrent_renorm_interval_overrides"] = {
            str(layer_id): int(interval) for layer_id, interval in sorted(recurrent_renorm_interval_overrides.items())
        }
    if readout_recurrent_renorm_interval_overrides:
        common_record["deltanet_statecache_readout_recurrent_renorm_interval_overrides"] = {
            str(layer_id): int(interval)
            for layer_id, interval in sorted(readout_recurrent_renorm_interval_overrides.items())
        }
    if args.paired_recurrent_mode_policy is not None or paired_recurrent_mode_overrides:
        common_record["paired_candidate_label"] = str(args.paired_label)
        common_record["paired_recurrent_mode_policy"] = args.paired_recurrent_mode_policy
        common_record["paired_recurrent_mode_overrides"] = {
            str(layer_id): mode for layer_id, mode in sorted(paired_recurrent_mode_overrides.items())
        }
        common_record["paired_order_schedule"] = str(args.paired_order_schedule)
    if args.paired_recurrent_group_size_policy is not None:
        common_record["paired_recurrent_group_size_policy"] = str(args.paired_recurrent_group_size_policy)
    if paired_recurrent_layer_group_size_overrides:
        common_record["paired_recurrent_layer_group_size_overrides"] = {
            str(layer_id): int(group) for layer_id, group in sorted(paired_recurrent_layer_group_size_overrides.items())
        }

    use_paired_case = (
        args.paired_recurrent_mode_policy is not None
        or bool(paired_recurrent_mode_overrides)
        or args.paired_recurrent_group_size_policy is not None
        or bool(paired_recurrent_layer_group_size_overrides)
    )

    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        if use_paired_case:
            _run_paired_case(
                harness,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                group_size=args.group_size,
                bits=args.bits,
                layer_bits_overrides=layer_bit_overrides,
                recurrent_group_size_policy=args.recurrent_group_size_policy,
                recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
                state_stage=args.state_stage,
                renorm_interval=args.renorm_interval,
                recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
                paired_recurrent_group_size_policy=args.paired_recurrent_group_size_policy,
                paired_recurrent_layer_group_size_overrides=paired_recurrent_layer_group_size_overrides,
                baseline_recurrent_mode_policy=args.recurrent_mode_policy,
                baseline_recurrent_mode_overrides=recurrent_mode_overrides,
                paired_recurrent_mode_policy=args.paired_recurrent_mode_policy,
                paired_recurrent_mode_overrides=paired_recurrent_mode_overrides,
                readout_recurrent_policy=args.readout_recurrent_policy,
                readout_recurrent_mode_policy=args.readout_recurrent_mode_policy,
                readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
                readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
                warmup_in_process_repeats=args.warmup_in_process_repeats,
                in_process_repeats=args.in_process_repeats,
                paired_label=args.paired_label,
                paired_order_schedule=args.paired_order_schedule,
                base_record={**common_record, "prompt_mode": "repeat_count", "repeat_count": repeat_count},
                continue_on_error=args.continue_on_error,
            )
        else:
            _run_case(
                harness,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                group_size=args.group_size,
                bits=args.bits,
                layer_bits_overrides=layer_bit_overrides,
                recurrent_group_size_policy=args.recurrent_group_size_policy,
                recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
                state_stage=args.state_stage,
                renorm_interval=args.renorm_interval,
                recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
                recurrent_mode_policy=args.recurrent_mode_policy,
                recurrent_mode_overrides=recurrent_mode_overrides,
                readout_recurrent_policy=args.readout_recurrent_policy,
                readout_recurrent_mode_policy=args.readout_recurrent_mode_policy,
                readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
                readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
                warmup_in_process_repeats=args.warmup_in_process_repeats,
                in_process_repeats=args.in_process_repeats,
                base_record={**common_record, "prompt_mode": "repeat_count", "repeat_count": repeat_count},
                continue_on_error=args.continue_on_error,
            )

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        if use_paired_case:
            _run_paired_case(
                harness,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                group_size=args.group_size,
                bits=args.bits,
                layer_bits_overrides=layer_bit_overrides,
                recurrent_group_size_policy=args.recurrent_group_size_policy,
                recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
                state_stage=args.state_stage,
                renorm_interval=args.renorm_interval,
                recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
                paired_recurrent_group_size_policy=args.paired_recurrent_group_size_policy,
                paired_recurrent_layer_group_size_overrides=paired_recurrent_layer_group_size_overrides,
                baseline_recurrent_mode_policy=args.recurrent_mode_policy,
                baseline_recurrent_mode_overrides=recurrent_mode_overrides,
                paired_recurrent_mode_policy=args.paired_recurrent_mode_policy,
                paired_recurrent_mode_overrides=paired_recurrent_mode_overrides,
                readout_recurrent_policy=args.readout_recurrent_policy,
                readout_recurrent_mode_policy=args.readout_recurrent_mode_policy,
                readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
                readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
                warmup_in_process_repeats=args.warmup_in_process_repeats,
                in_process_repeats=args.in_process_repeats,
                paired_label=args.paired_label,
                paired_order_schedule=args.paired_order_schedule,
                base_record={**common_record, "prompt_mode": "exact_length", "prompt_length": prompt_length},
                continue_on_error=args.continue_on_error,
            )
        else:
            _run_case(
                harness,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                group_size=args.group_size,
                bits=args.bits,
                layer_bits_overrides=layer_bit_overrides,
                recurrent_group_size_policy=args.recurrent_group_size_policy,
                recurrent_layer_group_size_overrides=recurrent_layer_group_size_overrides,
                state_stage=args.state_stage,
                renorm_interval=args.renorm_interval,
                recurrent_renorm_interval_overrides=recurrent_renorm_interval_overrides,
                recurrent_mode_policy=args.recurrent_mode_policy,
                recurrent_mode_overrides=recurrent_mode_overrides,
                readout_recurrent_policy=args.readout_recurrent_policy,
                readout_recurrent_mode_policy=args.readout_recurrent_mode_policy,
                readout_recurrent_renorm_interval_overrides=readout_recurrent_renorm_interval_overrides,
                readout_recurrent_mode_overrides=readout_recurrent_mode_overrides,
                warmup_in_process_repeats=args.warmup_in_process_repeats,
                in_process_repeats=args.in_process_repeats,
                base_record={**common_record, "prompt_mode": "exact_length", "prompt_length": prompt_length},
                continue_on_error=args.continue_on_error,
            )


if __name__ == "__main__":
    main()
