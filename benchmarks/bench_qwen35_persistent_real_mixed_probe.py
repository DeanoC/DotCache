from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _persistent_base_config,
    _resolve_prompt_records,
)
from dotcache.config import DotCacheConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    load_qwen35_text_only_from_pretrained,
    run_qwen35_attention_subset_persistent_serving_harness,
    transformers_available,
)


_DEFAULT_REAL_MIXED_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_serving_policy_compare_20260411_large_promptfiles_uncapped_longdecode_stage9_metal_packed_streaming_ci16/qwen35_persistent_serving_policy_compare.json"
)
_REAL_MIXED_KEY_CENTROIDS = {19: 8, 23: 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical real-mixed Stage 9 probe.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default=str(_DEFAULT_REAL_MIXED_MANIFEST))
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--detailed-timing", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def real_mixed_probe_dotcache_config() -> DotCacheConfig:
    return DotCacheConfig(
        head_dim=256,
        group_size=32,
        bits_k=8,
        bits_v=4,
        tokens_per_page=16,
    )


def real_mixed_probe_serving_config(*, policy_path: str | None, detailed_timing: bool = False) -> Any:
    config = _persistent_base_config(
        policy_path=policy_path,
        enable_early_exit=True,
        full_attention_check_interval=16,
        full_attention_streaming_order_mode="priority_value_hybrid",
        full_attention_streaming_priority_value_upper_weight=0.25,
        full_attention_key_centroid_count_by_layer=dict(_REAL_MIXED_KEY_CENTROIDS),
        enable_mixed_execution=True,
        mixed_execution_strategy="direct_m0",
        allow_value_m0=False,
        max_k_comp_error=0.20,
    )
    config.full_attention_mixed_mode_detailed_timing = bool(detailed_timing)
    return config


def _metric_sum(result: dict[str, Any], key: str) -> float:
    return float(sum(float(value) for value in result.get(key, {}).values()))


def _metric_sum_int(result: dict[str, Any], key: str) -> int:
    return int(sum(int(value) for value in result.get(key, {}).values()))


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "case_count": 0,
            "hand_tuned_avg_ms_per_step": 0.0,
            "bias_avg_ms_per_step": 0.0,
            "bias_vs_hand_exact_match_rate": 0.0,
            "bias_beats_hand_tuned_latency_rate": 0.0,
            "hand_tuned_direct_m0_query_prep_ms_per_case": 0.0,
            "bias_direct_m0_query_prep_ms_per_case": 0.0,
            "hand_tuned_direct_m0_gather_ms_per_case": 0.0,
            "bias_direct_m0_gather_ms_per_case": 0.0,
            "hand_tuned_direct_m0_score_ms_per_case": 0.0,
            "bias_direct_m0_score_ms_per_case": 0.0,
            "hand_tuned_exact_m3_score_ms_per_case": 0.0,
            "bias_exact_m3_score_ms_per_case": 0.0,
            "hand_tuned_final_mix_ms_per_case": 0.0,
            "bias_final_mix_ms_per_case": 0.0,
            "hand_tuned_final_mix_logits_ms_per_case": 0.0,
            "bias_final_mix_logits_ms_per_case": 0.0,
            "hand_tuned_final_mix_softmax_ms_per_case": 0.0,
            "bias_final_mix_softmax_ms_per_case": 0.0,
            "hand_tuned_final_mix_value_ms_per_case": 0.0,
            "bias_final_mix_value_ms_per_case": 0.0,
            "hand_tuned_executed_m0_blocks_per_case": 0.0,
            "bias_executed_m0_blocks_per_case": 0.0,
            "hand_tuned_executed_m3_blocks_per_case": 0.0,
            "bias_executed_m3_blocks_per_case": 0.0,
        }
    case_count = len(records)
    return {
        "case_count": int(case_count),
        "hand_tuned_avg_ms_per_step": float(
            sum(float(record["hand_tuned_decode_ms_per_step"]) for record in records) / case_count
        ),
        "bias_avg_ms_per_step": float(sum(float(record["bias_decode_ms_per_step"]) for record in records) / case_count),
        "bias_vs_hand_exact_match_rate": float(
            sum(int(bool(record["bias_matches_hand_tuned_exact"])) for record in records) / case_count
        ),
        "bias_beats_hand_tuned_latency_rate": float(
            sum(
                int(float(record["bias_decode_ms_per_step"]) < float(record["hand_tuned_decode_ms_per_step"]))
                for record in records
            )
            / case_count
        ),
        "hand_tuned_direct_m0_query_prep_ms_per_case": float(
            sum(float(record["hand_tuned_direct_m0_query_prep_ms_total"]) for record in records) / case_count
        ),
        "bias_direct_m0_query_prep_ms_per_case": float(
            sum(float(record["bias_direct_m0_query_prep_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_direct_m0_gather_ms_per_case": float(
            sum(float(record["hand_tuned_direct_m0_gather_ms_total"]) for record in records) / case_count
        ),
        "bias_direct_m0_gather_ms_per_case": float(
            sum(float(record["bias_direct_m0_gather_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_direct_m0_score_ms_per_case": float(
            sum(float(record["hand_tuned_direct_m0_score_ms_total"]) for record in records) / case_count
        ),
        "bias_direct_m0_score_ms_per_case": float(
            sum(float(record["bias_direct_m0_score_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_exact_m3_score_ms_per_case": float(
            sum(float(record["hand_tuned_exact_m3_score_ms_total"]) for record in records) / case_count
        ),
        "bias_exact_m3_score_ms_per_case": float(
            sum(float(record["bias_exact_m3_score_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_final_mix_ms_per_case": float(
            sum(float(record["hand_tuned_final_mix_ms_total"]) for record in records) / case_count
        ),
        "bias_final_mix_ms_per_case": float(
            sum(float(record["bias_final_mix_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_final_mix_logits_ms_per_case": float(
            sum(float(record["hand_tuned_final_mix_logits_ms_total"]) for record in records) / case_count
        ),
        "bias_final_mix_logits_ms_per_case": float(
            sum(float(record["bias_final_mix_logits_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_final_mix_softmax_ms_per_case": float(
            sum(float(record["hand_tuned_final_mix_softmax_ms_total"]) for record in records) / case_count
        ),
        "bias_final_mix_softmax_ms_per_case": float(
            sum(float(record["bias_final_mix_softmax_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_final_mix_value_ms_per_case": float(
            sum(float(record["hand_tuned_final_mix_value_ms_total"]) for record in records) / case_count
        ),
        "bias_final_mix_value_ms_per_case": float(
            sum(float(record["bias_final_mix_value_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_executed_m0_blocks_per_case": float(
            sum(float(record["hand_tuned_executed_m0_block_count_total"]) for record in records) / case_count
        ),
        "bias_executed_m0_blocks_per_case": float(
            sum(float(record["bias_executed_m0_block_count_total"]) for record in records) / case_count
        ),
        "hand_tuned_executed_m3_blocks_per_case": float(
            sum(float(record["hand_tuned_executed_m3_block_count_total"]) for record in records) / case_count
        ),
        "bias_executed_m3_blocks_per_case": float(
            sum(float(record["bias_executed_m3_block_count_total"]) for record in records) / case_count
        ),
    }


def _render_markdown(*, payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Qwen3.5 Real Mixed Probe",
        "",
        "## Summary",
        "",
        f"- case count: {int(summary['case_count'])}",
        f"- hand-tuned avg ms/step: {float(summary['hand_tuned_avg_ms_per_step']):.4f}",
        f"- bias avg ms/step: {float(summary['bias_avg_ms_per_step']):.4f}",
        f"- bias vs hand exact match rate: {float(summary['bias_vs_hand_exact_match_rate']):.3f}",
        f"- bias beats hand-tuned latency rate: {float(summary['bias_beats_hand_tuned_latency_rate']):.3f}",
        f"- hand-tuned direct-M0 query-prep ms/case: {float(summary['hand_tuned_direct_m0_query_prep_ms_per_case']):.4f}",
        f"- bias direct-M0 query-prep ms/case: {float(summary['bias_direct_m0_query_prep_ms_per_case']):.4f}",
        f"- hand-tuned direct-M0 gather ms/case: {float(summary['hand_tuned_direct_m0_gather_ms_per_case']):.4f}",
        f"- bias direct-M0 gather ms/case: {float(summary['bias_direct_m0_gather_ms_per_case']):.4f}",
        f"- hand-tuned direct-M0 score ms/case: {float(summary['hand_tuned_direct_m0_score_ms_per_case']):.4f}",
        f"- bias direct-M0 score ms/case: {float(summary['bias_direct_m0_score_ms_per_case']):.4f}",
        f"- hand-tuned exact-M3 score ms/case: {float(summary['hand_tuned_exact_m3_score_ms_per_case']):.4f}",
        f"- bias exact-M3 score ms/case: {float(summary['bias_exact_m3_score_ms_per_case']):.4f}",
        f"- hand-tuned final-mix ms/case: {float(summary['hand_tuned_final_mix_ms_per_case']):.4f}",
        f"- bias final-mix ms/case: {float(summary['bias_final_mix_ms_per_case']):.4f}",
        f"- hand-tuned final-mix logits ms/case: {float(summary['hand_tuned_final_mix_logits_ms_per_case']):.4f}",
        f"- bias final-mix logits ms/case: {float(summary['bias_final_mix_logits_ms_per_case']):.4f}",
        f"- hand-tuned final-mix softmax ms/case: {float(summary['hand_tuned_final_mix_softmax_ms_per_case']):.4f}",
        f"- bias final-mix softmax ms/case: {float(summary['bias_final_mix_softmax_ms_per_case']):.4f}",
        f"- hand-tuned final-mix value ms/case: {float(summary['hand_tuned_final_mix_value_ms_per_case']):.4f}",
        f"- bias final-mix value ms/case: {float(summary['bias_final_mix_value_ms_per_case']):.4f}",
        f"- hand-tuned executed M0 blocks/case: {float(summary['hand_tuned_executed_m0_blocks_per_case']):.2f}",
        f"- bias executed M0 blocks/case: {float(summary['bias_executed_m0_blocks_per_case']):.2f}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise RuntimeError("transformers is required for the real mixed probe")

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
        persistent_serving_config=real_mixed_probe_serving_config(
            policy_path=str(_DEFAULT_POLICY_PATH),
            detailed_timing=bool(args.detailed_timing),
        ),
        backend=str(args.backend),
    )

    records: list[dict[str, Any]] = []
    for prompt_record in prompt_records:
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
            policy_path=None,
            detailed_timing=bool(args.detailed_timing),
        )
        hand_result = run_qwen35_attention_subset_persistent_serving_harness(
            model,
            adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        adapter.persistent_serving_config = real_mixed_probe_serving_config(
            policy_path=str(_DEFAULT_POLICY_PATH),
            detailed_timing=bool(args.detailed_timing),
        )
        bias_result = run_qwen35_attention_subset_persistent_serving_harness(
            model,
            adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        records.append(
            {
                "case_tag": str(prompt_record["case_tag"]),
                "prompt_file_path": str(prompt_path),
                "prompt_length": int(prompt_record.get("prompt_length", 0)),
                "hand_tuned_decode_ms_per_step": float(hand_result.get("persistent_decode_ms_per_step", 0.0)),
                "bias_decode_ms_per_step": float(bias_result.get("persistent_decode_ms_per_step", 0.0)),
                "bias_matches_hand_tuned_exact": (
                    [int(token_id) for token_id in bias_result.get("persistent_generated_ids", [])]
                    == [int(token_id) for token_id in hand_result.get("persistent_generated_ids", [])]
                ),
                "hand_tuned_direct_m0_query_prep_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_direct_m0_query_prep_ms_total_by_layer",
                ),
                "bias_direct_m0_query_prep_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_direct_m0_query_prep_ms_total_by_layer",
                ),
                "hand_tuned_direct_m0_gather_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_direct_m0_gather_ms_total_by_layer",
                ),
                "bias_direct_m0_gather_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_direct_m0_gather_ms_total_by_layer",
                ),
                "hand_tuned_direct_m0_score_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_direct_m0_score_ms_total_by_layer",
                ),
                "bias_direct_m0_score_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_direct_m0_score_ms_total_by_layer",
                ),
                "hand_tuned_exact_m3_score_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_exact_m3_score_ms_total_by_layer",
                ),
                "bias_exact_m3_score_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_exact_m3_score_ms_total_by_layer",
                ),
                "hand_tuned_final_mix_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_final_mix_ms_total_by_layer",
                ),
                "bias_final_mix_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_final_mix_ms_total_by_layer",
                ),
                "hand_tuned_final_mix_logits_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_final_mix_logits_ms_total_by_layer",
                ),
                "bias_final_mix_logits_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_final_mix_logits_ms_total_by_layer",
                ),
                "hand_tuned_final_mix_softmax_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_final_mix_softmax_ms_total_by_layer",
                ),
                "bias_final_mix_softmax_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_final_mix_softmax_ms_total_by_layer",
                ),
                "hand_tuned_final_mix_value_ms_total": _metric_sum(
                    hand_result,
                    "persistent_full_attention_mixed_execution_final_mix_value_ms_total_by_layer",
                ),
                "bias_final_mix_value_ms_total": _metric_sum(
                    bias_result,
                    "persistent_full_attention_mixed_execution_final_mix_value_ms_total_by_layer",
                ),
                "hand_tuned_executed_m0_block_count_total": _metric_sum_int(
                    hand_result,
                    "persistent_full_attention_executed_m0_block_count_total_by_layer",
                ),
                "bias_executed_m0_block_count_total": _metric_sum_int(
                    bias_result,
                    "persistent_full_attention_executed_m0_block_count_total_by_layer",
                ),
                "hand_tuned_executed_m3_block_count_total": _metric_sum_int(
                    hand_result,
                    "persistent_full_attention_executed_m3_block_count_total_by_layer",
                ),
                "bias_executed_m3_block_count_total": _metric_sum_int(
                    bias_result,
                    "persistent_full_attention_executed_m3_block_count_total_by_layer",
                ),
            }
        )

    payload = {
        "config": {
            "model_id": str(args.model_id),
            "device": str(args.device) if args.device else None,
            "backend": str(args.backend),
            "torch_dtype": str(args.torch_dtype),
            "weight_quantization": str(args.weight_quantization),
            "decode_steps": int(args.decode_steps),
            "manifest_path": str(args.manifest_path) if args.manifest_path else None,
            "prompt_file_count": len(prompt_records),
            "bits_k": 8,
            "bits_v": 4,
            "tokens_per_page": 16,
            "group_size": 32,
            "mixed_execution_strategy": "direct_m0",
            "mixed_execution_max_k_comp_error": 0.20,
            "enable_early_exit": True,
            "full_attention_check_interval": 16,
            "full_attention_streaming_order_mode": "priority_value_hybrid",
            "full_attention_streaming_priority_value_upper_weight": 0.25,
            "full_attention_key_centroid_count_by_layer": dict(_REAL_MIXED_KEY_CENTROIDS),
            "full_attention_mixed_mode_detailed_timing": bool(args.detailed_timing),
        },
        "records": records,
        "summary": _summarize_records(records),
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(_render_markdown(payload=payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
