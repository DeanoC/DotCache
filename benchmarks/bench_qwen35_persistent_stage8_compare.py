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
    run_qwen35_persistent_full_attention_snapshot_comparison,
    run_qwen35_text_generation_harness,
    transformers_available,
)


_DEFAULT_CORPUS_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_shortlist_runtime_external_validation_20260410_4096_promptfiles_longdecode/corpus_manifest.json"
)
_DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_shortlist_runtime_candidate_20260410_4096_nonsynthetic_longdecode/persistent_shortlist_policy.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 8 conservative persistent DotCache integration on replay and serving.")
    parser.add_argument("--corpus-manifest-path", default=str(_DEFAULT_CORPUS_MANIFEST))
    parser.add_argument("--shortlist-policy-path", default=str(_DEFAULT_POLICY_PATH))
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


def _matching_prefix_length(reference_ids: list[int], candidate_ids: list[int]) -> int:
    prefix = 0
    for reference_id, candidate_id in zip(reference_ids, candidate_ids):
        if int(reference_id) != int(candidate_id):
            break
        prefix += 1
    return int(prefix)


def _sum_metric_by_layer(payload: dict[str, Any], key: str) -> float:
    return float(sum(float(value) for value in payload.get(key, {}).values()))


def _resolve_prompt_records_from_corpus_manifest(corpus_manifest_path: str) -> list[dict[str, Any]]:
    manifest = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
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
    return sorted(records, key=lambda record: str(record["case_tag"]))


def _resolve_snapshot_records_from_corpus_manifest(corpus_manifest_path: str) -> list[dict[str, Any]]:
    manifest = json.loads(Path(corpus_manifest_path).read_text(encoding="utf-8"))
    snapshot_records: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        child_manifest_path = record.get("paged_attention_snapshot_corpus_manifest_path")
        if not child_manifest_path:
            continue
        child_manifest = json.loads(Path(child_manifest_path).read_text(encoding="utf-8"))
        case_tag = str(record.get("case_tag", Path(str(child_manifest_path)).parent.name))
        for child_record in child_manifest.get("snapshot_records", []):
            snapshot_records.append(
                {
                    "case_tag": case_tag,
                    "snapshot_path": str(Path(child_record["paged_attention_snapshot_path"]).resolve()),
                    "layer_id": int(child_record.get("paged_attention_snapshot_layer_id", 0)),
                    "kv_head_id": int(child_record.get("paged_attention_snapshot_kv_head_id", 0)),
                    "step_index": int(child_record.get("paged_attention_snapshot_step_index", 0)),
                }
            )
    return sorted(
        snapshot_records,
        key=lambda record: (
            str(record["case_tag"]),
            int(record["layer_id"]),
            int(record["kv_head_id"]),
            int(record["step_index"]),
            str(record["snapshot_path"]),
        ),
    )


def _persistent_base_config(*, enable_compression: bool, policy_path: str | None) -> PersistentServingConfig:
    return PersistentServingConfig(
        enable_priority=True,
        enable_compression=bool(enable_compression),
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


def _build_stage8_dotcache_config(
    *,
    head_dim: int,
    group_size: int,
    bits_k: int,
    bits_v: int,
    tokens_per_page: int,
    enable_compression: bool,
) -> DotCacheConfig:
    return DotCacheConfig(
        head_dim=int(head_dim),
        group_size=int(group_size),
        bits_k=int(bits_k),
        bits_v=int(bits_v),
        tokens_per_page=int(tokens_per_page),
        default_mode_k="M0" if bool(enable_compression) else "M3",
        default_mode_v="M3",
    )


def _summarize_replay_pair_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "snapshot_count": 0,
            "selection_changed_rate": 0.0,
            "baseline_avg_max_abs_error": 0.0,
            "stage8_avg_max_abs_error": 0.0,
            "stage8_avg_selected_token_count": 0.0,
            "stage8_avg_selected_m0_metadata_block_count": 0.0,
            "stage8_avg_compression_invalid_block_count": 0.0,
        }
    count = len(records)
    return {
        "snapshot_count": int(count),
        "selection_changed_rate": float(
            sum(int(bool(record["selection_changed"])) for record in records) / count
        ),
        "baseline_avg_max_abs_error": float(
            sum(float(record["baseline_max_abs_error"]) for record in records) / count
        ),
        "stage8_avg_max_abs_error": float(
            sum(float(record["stage8_max_abs_error"]) for record in records) / count
        ),
        "baseline_max_abs_error": float(max(float(record["baseline_max_abs_error"]) for record in records)),
        "stage8_max_abs_error": float(max(float(record["stage8_max_abs_error"]) for record in records)),
        "baseline_avg_selected_token_count": float(
            sum(float(record["baseline_selected_token_count"]) for record in records) / count
        ),
        "stage8_avg_selected_token_count": float(
            sum(float(record["stage8_selected_token_count"]) for record in records) / count
        ),
        "stage8_avg_selected_m0_metadata_block_count": float(
            sum(float(record["stage8_selected_m0_metadata_block_count"]) for record in records) / count
        ),
        "stage8_avg_compression_invalid_block_count": float(
            sum(float(record["stage8_compression_invalid_block_count"]) for record in records) / count
        ),
        "stage8_avg_metadata_m0_block_count": float(
            sum(float(record["stage8_metadata_m0_block_count"]) for record in records) / count
        ),
    }


def _summarize_serving_pair_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "case_count": 0,
            "dense_avg_ms_per_step": 0.0,
            "baseline_avg_ms_per_step": 0.0,
            "stage8_avg_ms_per_step": 0.0,
            "stage8_beats_baseline_latency_rate": 0.0,
            "stage8_matches_baseline_exact_rate": 0.0,
            "stage8_matches_dense_exact_rate": 0.0,
            "stage8_avg_selected_m0_metadata_block_count": 0.0,
            "stage8_avg_dense_fallback_count": 0.0,
        }
    count = len(records)
    return {
        "case_count": int(count),
        "dense_avg_ms_per_step": float(sum(float(record["dense_decode_ms_per_step"]) for record in records) / count),
        "baseline_avg_ms_per_step": float(sum(float(record["baseline_decode_ms_per_step"]) for record in records) / count),
        "stage8_avg_ms_per_step": float(sum(float(record["stage8_decode_ms_per_step"]) for record in records) / count),
        "baseline_matches_dense_exact_rate": float(
            sum(int(bool(record["baseline_matches_dense_exact"])) for record in records) / count
        ),
        "stage8_matches_dense_exact_rate": float(
            sum(int(bool(record["stage8_matches_dense_exact"])) for record in records) / count
        ),
        "stage8_matches_baseline_exact_rate": float(
            sum(int(bool(record["stage8_matches_baseline_exact"])) for record in records) / count
        ),
        "stage8_beats_baseline_latency_rate": float(
            sum(
                int(float(record["stage8_decode_ms_per_step"]) < float(record["baseline_decode_ms_per_step"]))
                for record in records
            )
            / count
        ),
        "stage8_avg_selected_m0_metadata_block_count": float(
            sum(float(record["stage8_selected_m0_metadata_block_count_total"]) for record in records) / count
        ),
        "stage8_avg_dense_fallback_count": float(
            sum(float(record["stage8_dense_fallback_count_total"]) for record in records) / count
        ),
        "stage8_avg_compression_rerank_count": float(
            sum(float(record["stage8_compression_rerank_count_total"]) for record in records) / count
        ),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    replay_summary = payload["replay"]["summary"]
    serving_summary = payload["serving"]["summary"]
    lines: list[str] = []
    lines.append("# Qwen3.5 Persistent Stage 8 Conservative Validation\n")
    lines.append("## Replay Summary\n")
    lines.append(f"- snapshot count: {int(replay_summary['snapshot_count'])}")
    lines.append(f"- selection changed rate: {float(replay_summary['selection_changed_rate']):.3f}")
    lines.append(f"- baseline avg max abs error: {float(replay_summary['baseline_avg_max_abs_error']):.6f}")
    lines.append(f"- stage8 avg max abs error: {float(replay_summary['stage8_avg_max_abs_error']):.6f}")
    lines.append(f"- baseline max abs error: {float(replay_summary['baseline_max_abs_error']):.6f}")
    lines.append(f"- stage8 max abs error: {float(replay_summary['stage8_max_abs_error']):.6f}")
    lines.append(
        f"- stage8 avg selected M0-metadata blocks: {float(replay_summary['stage8_avg_selected_m0_metadata_block_count']):.3f}"
    )
    lines.append(
        f"- stage8 avg compression-invalid blocks: {float(replay_summary['stage8_avg_compression_invalid_block_count']):.3f}"
    )
    lines.append("\n## Serving Summary\n")
    lines.append(f"- case count: {int(serving_summary['case_count'])}")
    lines.append(f"- dense avg ms/step: {float(serving_summary['dense_avg_ms_per_step']):.4f}")
    lines.append(f"- baseline avg ms/step: {float(serving_summary['baseline_avg_ms_per_step']):.4f}")
    lines.append(f"- stage8 avg ms/step: {float(serving_summary['stage8_avg_ms_per_step']):.4f}")
    lines.append(
        f"- stage8 faster than baseline rate: {float(serving_summary['stage8_beats_baseline_latency_rate']):.3f}"
    )
    lines.append(
        f"- stage8 matches baseline exact rate: {float(serving_summary['stage8_matches_baseline_exact_rate']):.3f}"
    )
    lines.append(
        f"- stage8 matches dense exact rate: {float(serving_summary['stage8_matches_dense_exact_rate']):.3f}"
    )
    lines.append(
        f"- stage8 avg selected M0-metadata blocks: {float(serving_summary['stage8_avg_selected_m0_metadata_block_count']):.3f}"
    )
    lines.append(f"- stage8 avg dense fallback count: {float(serving_summary['stage8_avg_dense_fallback_count']):.3f}")
    lines.append("\n## Read\n")
    lines.append(
        "Replay isolates whether compression-aware M0/M3 metadata changes ranking while still executing selected blocks exactly."
    )
    lines.append(
        "Serving checks the real persistent harness for generated-id parity, latency, and whether Stage 8 fallback/compression telemetry stays conservative."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    corpus_manifest_path = str(Path(args.corpus_manifest_path).resolve())
    prompt_records = _resolve_prompt_records_from_corpus_manifest(corpus_manifest_path)
    snapshot_records = _resolve_snapshot_records_from_corpus_manifest(corpus_manifest_path)
    if not prompt_records:
        raise SystemExit("no prompt records resolved from corpus manifest")
    if not snapshot_records:
        raise SystemExit("no replay snapshot records resolved from corpus manifest")
    if not transformers_available():
        raise SystemExit("bench_qwen35_persistent_stage8_compare.py requires the optional transformers dependencies")

    replay_records: list[dict[str, Any]] = []
    for snapshot_record in snapshot_records:
        snapshot_path = str(snapshot_record["snapshot_path"])
        baseline = run_qwen35_persistent_full_attention_snapshot_comparison(
            snapshot_path,
            persistent_serving_config=_persistent_base_config(enable_compression=False, policy_path=None),
            dotcache_config=_build_stage8_dotcache_config(
                head_dim=256,
                group_size=int(args.group_size),
                bits_k=int(args.bits_k),
                bits_v=int(args.bits_v),
                tokens_per_page=int(args.tokens_per_page),
                enable_compression=False,
            ),
            history_mode="none",
        )
        stage8 = run_qwen35_persistent_full_attention_snapshot_comparison(
            snapshot_path,
            persistent_serving_config=_persistent_base_config(enable_compression=True, policy_path=None),
            dotcache_config=_build_stage8_dotcache_config(
                head_dim=256,
                group_size=int(args.group_size),
                bits_k=int(args.bits_k),
                bits_v=int(args.bits_v),
                tokens_per_page=int(args.tokens_per_page),
                enable_compression=True,
            ),
            history_mode="none",
        )
        replay_record = {
            "case_tag": str(snapshot_record["case_tag"]),
            "snapshot_path": snapshot_path,
            "layer_id": int(snapshot_record["layer_id"]),
            "kv_head_id": int(snapshot_record["kv_head_id"]),
            "step_index": int(snapshot_record["step_index"]),
            "selection_changed": bool(stage8["optional_block_ids"] != baseline["optional_block_ids"]),
            "baseline_selected_token_count": int(baseline["selected_token_count"]),
            "stage8_selected_token_count": int(stage8["selected_token_count"]),
            "baseline_max_abs_error": float(baseline["max_abs_error"]),
            "stage8_max_abs_error": float(stage8["max_abs_error"]),
            "stage8_selected_m0_metadata_block_count": int(stage8["selected_m0_metadata_block_count"]),
            "stage8_selected_m3_metadata_block_count": int(stage8["selected_m3_metadata_block_count"]),
            "stage8_compression_invalid_block_count": int(stage8["compression_invalid_block_count"]),
            "stage8_metadata_m0_block_count": int(stage8["persistent_full_attention_m0_metadata_block_count"]),
            "stage8_metadata_m3_block_count": int(stage8["persistent_full_attention_m3_metadata_block_count"]),
            "stage8_fallback_recommended": bool(stage8["fallback_recommended"]),
            "stage8_beta_upper": float(stage8["beta_upper"]),
        }
        replay_records.append(replay_record)

    dense_model, dense_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    baseline_persistent_model, baseline_persistent_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    stage8_persistent_model, stage8_persistent_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        weight_quantization=args.weight_quantization,
    )
    dense_adapter = Qwen35TextModelAdapter(model=dense_model)
    baseline_adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=baseline_persistent_model,
        dotcache_config=_build_stage8_dotcache_config(
            head_dim=256,
            group_size=int(args.group_size),
            bits_k=int(args.bits_k),
            bits_v=int(args.bits_v),
            tokens_per_page=int(args.tokens_per_page),
            enable_compression=False,
        ),
        persistent_serving_config=_persistent_base_config(
            enable_compression=False,
            policy_path=str(args.shortlist_policy_path),
        ),
        backend=str(args.backend),
    )
    stage8_adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=stage8_persistent_model,
        dotcache_config=_build_stage8_dotcache_config(
            head_dim=256,
            group_size=int(args.group_size),
            bits_k=int(args.bits_k),
            bits_v=int(args.bits_v),
            tokens_per_page=int(args.tokens_per_page),
            enable_compression=True,
        ),
        persistent_serving_config=_persistent_base_config(
            enable_compression=True,
            policy_path=str(args.shortlist_policy_path),
        ),
        backend=str(args.backend),
    )
    serving_records: list[dict[str, Any]] = []
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
        baseline_persistent_device = next(baseline_persistent_model.parameters()).device
        baseline_persistent_input_ids, baseline_persistent_attention_mask = _build_prompt_text_inputs(
            baseline_persistent_tokenizer,
            device=baseline_persistent_device,
            prompt_text=prompt_text,
            prompt_length=int(prompt_record.get("prompt_length", 0)),
        )
        stage8_persistent_device = next(stage8_persistent_model.parameters()).device
        stage8_persistent_input_ids, stage8_persistent_attention_mask = _build_prompt_text_inputs(
            stage8_persistent_tokenizer,
            device=stage8_persistent_device,
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
        baseline_result = run_qwen35_attention_subset_persistent_serving_harness(
            baseline_persistent_model,
            baseline_adapter,
            input_ids=baseline_persistent_input_ids,
            attention_mask=baseline_persistent_attention_mask,
            tokenizer=baseline_persistent_tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        stage8_result = run_qwen35_attention_subset_persistent_serving_harness(
            stage8_persistent_model,
            stage8_adapter,
            input_ids=stage8_persistent_input_ids,
            attention_mask=stage8_persistent_attention_mask,
            tokenizer=stage8_persistent_tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        dense_ids = [int(token_id) for token_id in dense_result.get("dense_generated_ids", [])]
        baseline_ids = [int(token_id) for token_id in baseline_result.get("persistent_generated_ids", [])]
        stage8_ids = [int(token_id) for token_id in stage8_result.get("persistent_generated_ids", [])]
        serving_record = {
            "case_tag": str(prompt_record["case_tag"]),
            "prompt_file_path": str(prompt_path),
            "prompt_length": int(dense_input_ids.shape[1]),
            "decode_steps": int(args.decode_steps),
            "dense_generated_ids": dense_ids,
            "baseline_generated_ids": baseline_ids,
            "stage8_generated_ids": stage8_ids,
            "baseline_matches_dense_exact": bool(baseline_ids == dense_ids),
            "stage8_matches_dense_exact": bool(stage8_ids == dense_ids),
            "stage8_matches_baseline_exact": bool(stage8_ids == baseline_ids),
            "baseline_dense_prefix_match_length": _matching_prefix_length(dense_ids, baseline_ids),
            "stage8_dense_prefix_match_length": _matching_prefix_length(dense_ids, stage8_ids),
            "stage8_baseline_prefix_match_length": _matching_prefix_length(baseline_ids, stage8_ids),
            "dense_decode_ms_per_step": float(dense_result.get("dense_decode_ms_per_step", 0.0)),
            "baseline_decode_ms_per_step": float(baseline_result.get("persistent_decode_ms_per_step", 0.0)),
            "stage8_decode_ms_per_step": float(stage8_result.get("persistent_decode_ms_per_step", 0.0)),
            "baseline_selected_m0_metadata_block_count_total": _sum_metric_by_layer(
                baseline_result,
                "persistent_full_attention_selected_m0_metadata_block_count_total_by_layer",
            ),
            "stage8_selected_m0_metadata_block_count_total": _sum_metric_by_layer(
                stage8_result,
                "persistent_full_attention_selected_m0_metadata_block_count_total_by_layer",
            ),
            "baseline_dense_fallback_count_total": _sum_metric_by_layer(
                baseline_result,
                "persistent_full_attention_dense_fallback_count_by_layer",
            ),
            "stage8_dense_fallback_count_total": _sum_metric_by_layer(
                stage8_result,
                "persistent_full_attention_dense_fallback_count_by_layer",
            ),
            "baseline_compression_rerank_count_total": _sum_metric_by_layer(
                baseline_result,
                "persistent_full_attention_compression_rerank_count_by_layer",
            ),
            "stage8_compression_rerank_count_total": _sum_metric_by_layer(
                stage8_result,
                "persistent_full_attention_compression_rerank_count_by_layer",
            ),
            "stage8_enable_compression": bool(stage8_result.get("persistent_runtime_enable_compression", False)),
        }
        serving_records.append(serving_record)

    payload = {
        "config": {
            "corpus_manifest_path": corpus_manifest_path,
            "model_id": str(args.model_id),
            "device": args.device,
            "backend": str(args.backend),
            "torch_dtype": str(args.torch_dtype),
            "weight_quantization": str(args.weight_quantization),
            "tokens_per_page": int(args.tokens_per_page),
            "group_size": int(args.group_size),
            "bits_k": int(args.bits_k),
            "bits_v": int(args.bits_v),
            "decode_steps": int(args.decode_steps),
            "shortlist_policy_path": str(args.shortlist_policy_path),
        },
        "replay": {
            "records": replay_records,
            "summary": _summarize_replay_pair_records(replay_records),
        },
        "serving": {
            "records": serving_records,
            "summary": _summarize_serving_pair_records(serving_records),
        },
    }
    output_json = Path(args.output_json).resolve() if args.output_json else None
    output_md = Path(args.output_md).resolve() if args.output_md else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
