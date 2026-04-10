from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.config import DotCacheConfig
from dotcache.integrations.qwen35 import (
    PersistentServingConfig,
    Qwen35AttentionSubsetDotCacheModelAdapter,
    Qwen35TextModelAdapter,
    load_qwen35_text_only_from_pretrained,
    run_qwen35_attention_subset_persistent_serving_harness,
    run_qwen35_text_generation_harness,
    transformers_available,
)


_DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_shortlist_runtime_candidate_20260410_4096_nonsynthetic_longdecode/persistent_shortlist_policy.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen3.5 persistent serving hand-tuned vs bias policy guidance.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--shortlist-policy-path", default=str(_DEFAULT_POLICY_PATH))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _build_prompt_text_inputs(
    tokenizer: Any,
    *,
    device: Any,
    prompt_text: str,
    prompt_length: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("prompt text tokenized to an empty sequence")
    if prompt_length > 0:
        token_ids = token_ids[:prompt_length]
        if not token_ids:
            raise ValueError("prompt text does not contain enough tokens for the requested prompt_length")
    if tokenizer.bos_token_id is not None:
        token_ids = [int(tokenizer.bos_token_id)] + [int(token_id) for token_id in token_ids]
    else:
        token_ids = [int(token_id) for token_id in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _resolve_prompt_records(
    *,
    manifest_path: str | None,
    prompt_files: list[str],
    prompt_file_target_length: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for record in manifest.get("records", []):
            prompt_file_path = record.get("prompt_file_path")
            if not prompt_file_path:
                continue
            records.append(
                {
                    "case_tag": str(record.get("case_tag", Path(str(prompt_file_path)).stem)),
                    "prompt_file_path": str(Path(prompt_file_path).resolve()),
                    "prompt_length": int(record.get("prompt_length", 0)),
                }
            )
    for prompt_file in prompt_files:
        path = Path(prompt_file).resolve()
        records.append(
            {
                "case_tag": path.stem,
                "prompt_file_path": str(path),
                "prompt_length": max(int(prompt_file_target_length), 0),
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        deduped[str(record["prompt_file_path"])] = dict(record)
    return sorted(deduped.values(), key=lambda record: str(record["case_tag"]))


def _persistent_base_config(policy_path: str | None = None) -> PersistentServingConfig:
    return PersistentServingConfig(
        enable_priority=True,
        full_attention_sink_block_count=1,
        full_attention_recent_block_count=64,
        full_attention_mandatory_recent_block_count=16,
        full_attention_exploration_blocks_per_region=1,
        full_attention_optional_top_k=128,
        full_attention_optional_use_upper_bounds_first=False,
        full_attention_optional_upper_bound_quota=16,
        full_attention_optional_far_quota=32,
        full_attention_optional_mid_quota=48,
        full_attention_optional_near_quota=32,
        full_attention_optional_bootstrap_far_anchor_quota=4,
        full_attention_optional_far_anchor_quota=0,
        full_attention_optional_far_anchor_priority_margin=0.25,
        full_attention_optional_diversity_weight=0.5,
        full_attention_optional_diversity_radius=4,
        full_attention_optional_diversity_requires_history=True,
        full_attention_optional_diversity_min_history_count=1,
        full_attention_optional_diversity_max_history_count=2,
        full_attention_priority_prev_attention_weight=1.0,
        full_attention_priority_recency_weight=0.05,
        full_attention_priority_recency_decay_blocks=32.0,
        full_attention_priority_value_norm_weight=0.05,
        full_attention_shortlist_policy_path=policy_path,
    )


def _matching_prefix_length(reference_ids: list[int], candidate_ids: list[int]) -> int:
    prefix = 0
    for reference_id, candidate_id in zip(reference_ids, candidate_ids):
        if int(reference_id) != int(candidate_id):
            break
        prefix += 1
    return int(prefix)


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "case_count": 0,
            "dense_avg_ms_per_step": 0.0,
            "hand_tuned_avg_ms_per_step": 0.0,
            "bias_avg_ms_per_step": 0.0,
            "hand_vs_dense_exact_match_rate": 0.0,
            "bias_vs_dense_exact_match_rate": 0.0,
            "bias_vs_hand_exact_match_rate": 0.0,
            "bias_beats_hand_tuned_latency_rate": 0.0,
            "hand_tuned_policy_resolve_ms_per_case": 0.0,
            "bias_policy_resolve_ms_per_case": 0.0,
            "hand_tuned_score_ms_per_case": 0.0,
            "bias_score_ms_per_case": 0.0,
            "hand_tuned_selection_ms_per_case": 0.0,
            "bias_selection_ms_per_case": 0.0,
            "hand_tuned_policy_bias_ms_per_case": 0.0,
            "bias_policy_bias_ms_per_case": 0.0,
        }
    case_count = len(records)
    return {
        "case_count": int(case_count),
        "dense_avg_ms_per_step": float(sum(float(record["dense_decode_ms_per_step"]) for record in records) / case_count),
        "hand_tuned_avg_ms_per_step": float(
            sum(float(record["hand_tuned_decode_ms_per_step"]) for record in records) / case_count
        ),
        "bias_avg_ms_per_step": float(sum(float(record["bias_decode_ms_per_step"]) for record in records) / case_count),
        "hand_vs_dense_exact_match_rate": float(
            sum(int(bool(record["hand_tuned_matches_dense_exact"])) for record in records) / case_count
        ),
        "bias_vs_dense_exact_match_rate": float(
            sum(int(bool(record["bias_matches_dense_exact"])) for record in records) / case_count
        ),
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
        "hand_tuned_policy_resolve_ms_per_case": float(
            sum(float(record["hand_tuned_policy_resolve_ms_total"]) for record in records) / case_count
        ),
        "bias_policy_resolve_ms_per_case": float(
            sum(float(record["bias_policy_resolve_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_score_ms_per_case": float(
            sum(float(record["hand_tuned_score_ms_total"]) for record in records) / case_count
        ),
        "bias_score_ms_per_case": float(sum(float(record["bias_score_ms_total"]) for record in records) / case_count),
        "hand_tuned_selection_ms_per_case": float(
            sum(float(record["hand_tuned_selection_ms_total"]) for record in records) / case_count
        ),
        "bias_selection_ms_per_case": float(
            sum(float(record["bias_selection_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_policy_bias_ms_per_case": float(
            sum(float(record["hand_tuned_policy_bias_ms_total"]) for record in records) / case_count
        ),
        "bias_policy_bias_ms_per_case": float(
            sum(float(record["bias_policy_bias_ms_total"]) for record in records) / case_count
        ),
    }


def _render_markdown(*, records: list[dict[str, Any]], summary: dict[str, Any], payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3.5 Persistent Serving Policy Compare\n")
    lines.append("## Summary\n")
    lines.append(f"- case count: {int(summary['case_count'])}")
    lines.append(f"- dense avg ms/step: {float(summary['dense_avg_ms_per_step']):.4f}")
    lines.append(f"- hand-tuned avg ms/step: {float(summary['hand_tuned_avg_ms_per_step']):.4f}")
    lines.append(f"- bias avg ms/step: {float(summary['bias_avg_ms_per_step']):.4f}")
    lines.append(f"- hand-tuned vs dense exact match rate: {float(summary['hand_vs_dense_exact_match_rate']):.3f}")
    lines.append(f"- bias vs dense exact match rate: {float(summary['bias_vs_dense_exact_match_rate']):.3f}")
    lines.append(f"- bias vs hand-tuned exact match rate: {float(summary['bias_vs_hand_exact_match_rate']):.3f}")
    lines.append(f"- bias faster than hand-tuned rate: {float(summary['bias_beats_hand_tuned_latency_rate']):.3f}")
    lines.append(f"- hand-tuned policy resolve ms/case: {float(summary['hand_tuned_policy_resolve_ms_per_case']):.4f}")
    lines.append(f"- bias policy resolve ms/case: {float(summary['bias_policy_resolve_ms_per_case']):.4f}")
    lines.append(f"- hand-tuned score ms/case: {float(summary['hand_tuned_score_ms_per_case']):.4f}")
    lines.append(f"- bias score ms/case: {float(summary['bias_score_ms_per_case']):.4f}")
    lines.append(f"- hand-tuned selection ms/case: {float(summary['hand_tuned_selection_ms_per_case']):.4f}")
    lines.append(f"- bias selection ms/case: {float(summary['bias_selection_ms_per_case']):.4f}")
    lines.append(f"- hand-tuned policy-bias ms/case: {float(summary['hand_tuned_policy_bias_ms_per_case']):.4f}")
    lines.append(f"- bias policy-bias ms/case: {float(summary['bias_policy_bias_ms_per_case']):.4f}")
    lines.append("\n## Cases\n")
    for record in records:
        lines.append(
            f"- {record['case_tag']}: dense {float(record['dense_decode_ms_per_step']):.4f} ms/step, "
            f"hand {float(record['hand_tuned_decode_ms_per_step']):.4f}, bias {float(record['bias_decode_ms_per_step']):.4f}, "
            f"hand=dense {bool(record['hand_tuned_matches_dense_exact'])}, "
            f"bias=dense {bool(record['bias_matches_dense_exact'])}, "
            f"bias=hand {bool(record['bias_matches_hand_tuned_exact'])}, "
            f"hand select {float(record['hand_tuned_selection_ms_total']):.2f} ms, "
            f"bias select {float(record['bias_selection_ms_total']):.2f} ms"
        )
    lines.append("\n## Read\n")
    lines.append(
        "This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation."
    )
    lines.append(
        "It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_persistent_serving_policy_compare.py requires the optional transformers dependencies")
    prompt_records = _resolve_prompt_records(
        manifest_path=args.manifest_path,
        prompt_files=[str(path) for path in args.prompt_files],
        prompt_file_target_length=int(args.prompt_file_target_length),
    )
    if not prompt_records:
        raise SystemExit("no prompt records resolved; provide --manifest-path or --prompt-files")

    dense_model, dense_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    persistent_model, persistent_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    dotcache_config = DotCacheConfig(
        head_dim=256,
        group_size=int(args.group_size),
        bits_k=int(args.bits_k),
        bits_v=int(args.bits_v),
        tokens_per_page=int(args.tokens_per_page),
    )
    dense_adapter = Qwen35TextModelAdapter(model=dense_model)
    persistent_adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=persistent_model,
        dotcache_config=dotcache_config,
        persistent_serving_config=_persistent_base_config(policy_path=None),
        backend=str(args.backend),
    )
    records: list[dict[str, Any]] = []
    for prompt_record in prompt_records:
        prompt_path = Path(str(prompt_record["prompt_file_path"]))
        prompt_text = prompt_path.read_text(encoding="utf-8")
        dense_device = next(dense_model.parameters()).device
        dense_input_ids, dense_attention_mask = _build_prompt_text_inputs(
            dense_tokenizer,
            device=dense_device,
            prompt_text=prompt_text,
            prompt_length=int(prompt_record.get("prompt_length", 0)),
        )
        persistent_device = next(persistent_model.parameters()).device
        persistent_input_ids, persistent_attention_mask = _build_prompt_text_inputs(
            persistent_tokenizer,
            device=persistent_device,
            prompt_text=prompt_text,
            prompt_length=int(prompt_record.get("prompt_length", 0)),
        )
        dense_result = run_qwen35_text_generation_harness(
            dense_model,
            dense_adapter,
            input_ids=dense_input_ids,
            attention_mask=dense_attention_mask,
            max_new_tokens=int(args.decode_steps),
            tokenizer=dense_tokenizer,
        )
        persistent_adapter.persistent_serving_config = _persistent_base_config(policy_path=None)
        hand_result = run_qwen35_attention_subset_persistent_serving_harness(
            persistent_model,
            persistent_adapter,
            input_ids=persistent_input_ids,
            attention_mask=persistent_attention_mask,
            tokenizer=persistent_tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        persistent_adapter.persistent_serving_config = _persistent_base_config(policy_path=str(args.shortlist_policy_path))
        bias_result = run_qwen35_attention_subset_persistent_serving_harness(
            persistent_model,
            persistent_adapter,
            input_ids=persistent_input_ids,
            attention_mask=persistent_attention_mask,
            tokenizer=persistent_tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        dense_ids = [int(token_id) for token_id in dense_result.get("dense_generated_ids", [])]
        hand_ids = [int(token_id) for token_id in hand_result.get("persistent_generated_ids", [])]
        bias_ids = [int(token_id) for token_id in bias_result.get("persistent_generated_ids", [])]
        record = {
            "case_tag": str(prompt_record["case_tag"]),
            "prompt_file_path": str(prompt_path),
            "prompt_length": int(dense_input_ids.shape[1]),
            "decode_steps": int(args.decode_steps),
            "dense_generated_ids": dense_ids,
            "hand_tuned_generated_ids": hand_ids,
            "bias_generated_ids": bias_ids,
            "hand_tuned_matches_dense_exact": bool(hand_ids == dense_ids),
            "bias_matches_dense_exact": bool(bias_ids == dense_ids),
            "bias_matches_hand_tuned_exact": bool(bias_ids == hand_ids),
            "hand_tuned_dense_prefix_match_length": _matching_prefix_length(dense_ids, hand_ids),
            "bias_dense_prefix_match_length": _matching_prefix_length(dense_ids, bias_ids),
            "bias_hand_tuned_prefix_match_length": _matching_prefix_length(hand_ids, bias_ids),
            "dense_decode_ms_per_step": float(dense_result.get("dense_decode_ms_per_step", 0.0)),
            "hand_tuned_decode_ms_per_step": float(hand_result.get("persistent_decode_ms_per_step", 0.0)),
            "bias_decode_ms_per_step": float(bias_result.get("persistent_decode_ms_per_step", 0.0)),
            "hand_tuned_shortlist_policy_mode": str(hand_result.get("persistent_runtime_shortlist_policy_mode", "")),
            "bias_shortlist_policy_mode": str(bias_result.get("persistent_runtime_shortlist_policy_mode", "")),
            "hand_tuned_shortlist_policy_applied_count": int(
                hand_result.get("persistent_runtime_shortlist_policy_applied_count", 0)
            ),
            "bias_shortlist_policy_applied_count": int(
                bias_result.get("persistent_runtime_shortlist_policy_applied_count", 0)
            ),
            "hand_tuned_policy_resolve_ms_total": float(
                hand_result.get("persistent_shortlist_policy_resolve_ms_total", 0.0)
            ),
            "bias_policy_resolve_ms_total": float(
                bias_result.get("persistent_shortlist_policy_resolve_ms_total", 0.0)
            ),
            "hand_tuned_score_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get("persistent_full_attention_score_ms_total_by_layer", {}).values()
                )
            ),
            "bias_score_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get("persistent_full_attention_score_ms_total_by_layer", {}).values()
                )
            ),
            "hand_tuned_selection_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get("persistent_full_attention_selection_ms_total_by_layer", {}).values()
                )
            ),
            "bias_selection_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get("persistent_full_attention_selection_ms_total_by_layer", {}).values()
                )
            ),
            "hand_tuned_policy_bias_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get("persistent_full_attention_policy_bias_ms_total_by_layer", {}).values()
                )
            ),
            "bias_policy_bias_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get("persistent_full_attention_policy_bias_ms_total_by_layer", {}).values()
                )
            ),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = _summarize_records(records)
    payload = {
        "config": {
            "model_id": str(args.model_id),
            "device": args.device,
            "backend": str(args.backend),
            "torch_dtype": str(args.torch_dtype),
            "weight_quantization": str(args.weight_quantization),
            "decode_steps": int(args.decode_steps),
            "tokens_per_page": int(args.tokens_per_page),
            "shortlist_policy_path": str(args.shortlist_policy_path),
            "persistent_shortlist_policy_mode_default": "bias",
            "persistent_shortlist_policy_bias_weight_default": 0.10,
        },
        "records": records,
        "summary": summary,
    }
    output_json = Path(args.output_json).resolve() if args.output_json else None
    output_md = Path(args.output_md).resolve() if args.output_md else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(records=records, summary=summary, payload=payload), encoding="utf-8")


if __name__ == "__main__":
    main()
