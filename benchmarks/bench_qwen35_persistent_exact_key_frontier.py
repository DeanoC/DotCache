from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    _DEFAULT_REAL_MIXED_MANIFEST,
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _resolve_prompt_records,
)
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    load_qwen35_text_only_from_pretrained,
    run_qwen35_attention_subset_persistent_serving_harness,
    transformers_available,
)

_DEFAULT_FRONTIER_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_external_20260412.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study the exact-key fallback frontier for real-mixed Stage 9.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default=str(_DEFAULT_FRONTIER_MANIFEST))
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--target-layers", default=None)
    parser.add_argument("--sweep-thresholds", default='[0.20, 0.22, 0.24]')
    parser.add_argument("--warmup-runs-per-case", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _normalize_int_map(values: dict[str, Any] | dict[int, Any] | None) -> dict[int, float]:
    if not values:
        return {}
    return {int(key): float(value) for key, value in dict(values).items()}


def _run_bias_case(
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    prompt_record: dict[str, Any],
    decode_steps: int,
    max_k_comp_error_by_layer: dict[int, float] | None,
) -> dict[str, Any]:
    prompt_path = Path(str(prompt_record["prompt_file_path"]))
    prompt_text = prompt_path.read_text(encoding="utf-8")
    device = next(model.parameters()).device
    input_ids, attention_mask = _build_prompt_text_inputs(
        tokenizer,
        device=device,
        prompt_text=prompt_text,
        prompt_length=int(prompt_record.get("prompt_length", 0)),
    )
    adapter.persistent_serving_config = real_mixed_probe_serving_config(
        policy_path=str(_DEFAULT_POLICY_PATH),
        detailed_timing=False,
        max_k_comp_error_by_layer=max_k_comp_error_by_layer,
    )
    result = run_qwen35_attention_subset_persistent_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        decode_steps=int(decode_steps),
        persistent_policy_prompt_family=str(prompt_record["case_tag"]),
    )
    return {
        "case_tag": str(prompt_record["case_tag"]),
        "prompt_file_path": str(prompt_path),
        "prompt_length": int(prompt_record.get("prompt_length", 0)),
        "decode_ms_per_step": float(result.get("persistent_decode_ms_per_step", 0.0)),
        "generated_ids": [int(token_id) for token_id in result.get("persistent_generated_ids", [])],
        "executed_exact_key_m3_by_layer": _normalize_int_map(
            result.get("persistent_full_attention_executed_exact_key_m3_block_count_total_by_layer")
        ),
        "executed_m0_by_layer": _normalize_int_map(
            result.get("persistent_full_attention_executed_m0_block_count_total_by_layer")
        ),
        "direct_m0_gather_ms_by_layer": _normalize_int_map(
            result.get("persistent_full_attention_mixed_execution_direct_m0_gather_ms_total_by_layer")
        ),
        "direct_m0_score_ms_by_layer": _normalize_int_map(
            result.get("persistent_full_attention_mixed_execution_direct_m0_score_ms_total_by_layer")
        ),
        "exact_m3_score_ms_by_layer": _normalize_int_map(
            result.get("persistent_full_attention_mixed_execution_exact_m3_score_ms_total_by_layer")
        ),
    }


def _discover_candidate_layers(records: list[dict[str, Any]]) -> list[int]:
    layers: set[int] = set()
    for record in records:
        for layer_id, count in record.get("executed_exact_key_m3_by_layer", {}).items():
            if float(count) > 0.0:
                layers.add(int(layer_id))
    return sorted(layers)


def _average_layer_map(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not records:
        return {}
    all_layers = sorted({int(layer_id) for record in records for layer_id in record.get(key, {}).keys()})
    summary: dict[str, float] = {}
    for layer_id in all_layers:
        summary[str(layer_id)] = float(
            sum(float(record.get(key, {}).get(int(layer_id), 0.0)) for record in records) / len(records)
        )
    return summary


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "case_count": 0,
            "bias_avg_ms_per_step": 0.0,
            "executed_exact_key_m3_blocks_per_case": 0.0,
            "executed_exact_key_m3_by_layer_per_case": {},
            "executed_m0_by_layer_per_case": {},
            "direct_m0_gather_ms_per_case_by_layer": {},
            "direct_m0_score_ms_per_case_by_layer": {},
            "exact_m3_score_ms_per_case_by_layer": {},
        }
    return {
        "case_count": int(len(records)),
        "bias_avg_ms_per_step": float(sum(float(record["decode_ms_per_step"]) for record in records) / len(records)),
        "executed_exact_key_m3_blocks_per_case": float(
            sum(sum(float(value) for value in record.get("executed_exact_key_m3_by_layer", {}).values()) for record in records)
            / len(records)
        ),
        "executed_exact_key_m3_by_layer_per_case": _average_layer_map(records, "executed_exact_key_m3_by_layer"),
        "executed_m0_by_layer_per_case": _average_layer_map(records, "executed_m0_by_layer"),
        "direct_m0_gather_ms_per_case_by_layer": _average_layer_map(records, "direct_m0_gather_ms_by_layer"),
        "direct_m0_score_ms_per_case_by_layer": _average_layer_map(records, "direct_m0_score_ms_by_layer"),
        "exact_m3_score_ms_per_case_by_layer": _average_layer_map(records, "exact_m3_score_ms_by_layer"),
    }


def _summarize_sweep(
    *,
    baseline_records: list[dict[str, Any]],
    sweep_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_case = {str(record["case_tag"]): record for record in baseline_records}
    summary = _summarize_records(sweep_records)
    exact_match_rate = 0.0
    if sweep_records:
        exact_match_rate = float(
            sum(
                int(list(record["generated_ids"]) == list(baseline_by_case[str(record["case_tag"])]["generated_ids"]))
                for record in sweep_records
            )
            / len(sweep_records)
        )
    summary["bias_vs_baseline_exact_match_rate"] = float(exact_match_rate)
    summary["delta_vs_baseline_ms_per_step"] = float(
        summary["bias_avg_ms_per_step"] - _summarize_records(baseline_records)["bias_avg_ms_per_step"]
    )
    return summary


def _render_markdown(*, payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]["summary"]
    lines = [
        "# Qwen3.5 Exact-Key Frontier Study",
        "",
        "## Baseline",
        "",
        f"- case count: {int(baseline['case_count'])}",
        f"- bias avg ms/step: {float(baseline['bias_avg_ms_per_step']):.4f}",
        f"- executed exact-key M3 blocks/case: {float(baseline['executed_exact_key_m3_blocks_per_case']):.4f}",
        f"- candidate layers: {payload['candidate_layers']}",
        f"- executed exact-key M3 by layer/case: {json.dumps(baseline['executed_exact_key_m3_by_layer_per_case'], sort_keys=True)}",
        "",
        "## Sweeps",
        "",
    ]
    for sweep in payload["sweeps"]:
        summary = sweep["summary"]
        lines.extend(
            [
                f"- layer `{int(sweep['layer_id'])}` at threshold `{float(sweep['threshold']):.2f}`:",
                f"  - bias avg ms/step: {float(summary['bias_avg_ms_per_step']):.4f}",
                f"  - delta vs baseline ms/step: {float(summary['delta_vs_baseline_ms_per_step']):.4f}",
                f"  - bias vs baseline exact match rate: {float(summary['bias_vs_baseline_exact_match_rate']):.3f}",
                f"  - executed exact-key M3 blocks/case: {float(summary['executed_exact_key_m3_blocks_per_case']):.4f}",
                f"  - executed exact-key M3 by layer/case: {json.dumps(summary['executed_exact_key_m3_by_layer_per_case'], sort_keys=True)}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise RuntimeError("transformers is required for the exact-key frontier study")

    prompt_records = _resolve_prompt_records(
        manifest_path=str(args.manifest_path) if args.manifest_path else None,
        prompt_files=[str(value) for value in list(args.prompt_files)],
        prompt_file_target_length=int(args.prompt_file_target_length),
    )
    model, tokenizer = load_qwen35_text_only_from_pretrained(
        str(args.model_id),
        device=str(args.device) if args.device else None,
        torch_dtype=str(args.torch_dtype),
        weight_quantization=str(args.weight_quantization),
    )
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=real_mixed_probe_dotcache_config(),
        persistent_serving_config=real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH)),
        backend=str(args.backend),
    )

    warmup_runs_per_case = max(int(args.warmup_runs_per_case), 0)
    for _ in range(warmup_runs_per_case):
        for prompt_record in prompt_records:
            _run_bias_case(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                prompt_record=prompt_record,
                decode_steps=int(args.decode_steps),
                max_k_comp_error_by_layer=None,
            )
    baseline_records = [
        _run_bias_case(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            prompt_record=prompt_record,
            decode_steps=int(args.decode_steps),
            max_k_comp_error_by_layer=None,
        )
        for prompt_record in prompt_records
    ]
    if args.target_layers:
        candidate_layers = sorted(int(value) for value in json.loads(str(args.target_layers)))
    else:
        candidate_layers = _discover_candidate_layers(baseline_records)
    thresholds = [float(value) for value in json.loads(str(args.sweep_thresholds))]

    sweeps: list[dict[str, Any]] = []
    for layer_id in candidate_layers:
        for threshold in thresholds:
            override = {int(layer_id): float(threshold)}
            records = [
                _run_bias_case(
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    prompt_record=prompt_record,
                    decode_steps=int(args.decode_steps),
                    max_k_comp_error_by_layer=override,
                )
                for prompt_record in prompt_records
            ]
            sweeps.append(
                {
                    "layer_id": int(layer_id),
                    "threshold": float(threshold),
                    "override": {str(layer_id): float(threshold)},
                    "records": records,
                    "summary": _summarize_sweep(baseline_records=baseline_records, sweep_records=records),
                }
            )

    payload = {
        "config": {
            "model_id": str(args.model_id),
            "device": str(args.device),
            "backend": str(args.backend),
            "decode_steps": int(args.decode_steps),
            "manifest_path": str(args.manifest_path),
            "sweep_thresholds": thresholds,
            "candidate_layers": candidate_layers,
            "warmup_runs_per_case": warmup_runs_per_case,
        },
        "baseline": {
            "records": baseline_records,
            "summary": _summarize_records(baseline_records),
        },
        "candidate_layers": candidate_layers,
        "sweeps": sweeps,
    }

    if args.output_json:
        output_json_path = Path(str(args.output_json))
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        output_md_path = Path(str(args.output_md))
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(_render_markdown(payload=payload), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
