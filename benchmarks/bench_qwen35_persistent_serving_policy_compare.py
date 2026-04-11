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
    parser.add_argument("--enable-early-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-attention-check-interval", type=int, default=None)
    parser.add_argument("--full-attention-streaming-order-mode", default=None)
    parser.add_argument("--full-attention-streaming-proxy-value-weight-by-layer", default=None)
    parser.add_argument("--full-attention-streaming-priority-value-upper-weight", type=float, default=None)
    parser.add_argument("--full-attention-streaming-exact-value-rerank-layers", default=None)
    parser.add_argument("--full-attention-streaming-exact-value-rerank-max-remaining-blocks", type=int, default=None)
    parser.add_argument("--full-attention-refine-top-k", type=int, default=None)
    parser.add_argument("--full-attention-refine-top-k-by-layer", default=None)
    parser.add_argument("--full-attention-streaming-refine-top-k", type=int, default=None)
    parser.add_argument("--full-attention-probe-refine-top-k", type=int, default=None)
    parser.add_argument("--full-attention-probe-sample-count", type=int, default=None)
    parser.add_argument("--full-attention-mass-eps", type=float, default=None)
    parser.add_argument("--full-attention-value-eps", type=float, default=None)
    parser.add_argument("--full-attention-min-processed-blocks", type=int, default=None)
    parser.add_argument("--full-attention-region-residual-caps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--full-attention-residual-cluster-count", type=int, default=None)
    parser.add_argument("--full-attention-key-centroid-count", type=int, default=None)
    parser.add_argument("--full-attention-key-centroid-count-by-layer", default=None)
    parser.add_argument("--full-attention-value-centroid-count", type=int, default=None)
    parser.add_argument("--full-attention-value-centroid-count-by-layer", default=None)
    parser.add_argument("--enable-full-attention-mixed-mode-execution", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-full-attention-mixed-mode-execution-allow-value-m0", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--full-attention-mixed-mode-execution-strategy",
        choices=["cached_reconstruct", "blockwise_qdq", "direct_m0", "direct_m0_metal_packed"],
        default="cached_reconstruct",
    )
    parser.add_argument("--full-attention-mixed-mode-execution-max-k-comp-error", default="0.10")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--shortlist-policy-path", default=str(_DEFAULT_POLICY_PATH))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


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


def _persistent_base_config(
    policy_path: str | None = None,
    *,
    enable_early_exit: bool = False,
    full_attention_check_interval: int | None = None,
    full_attention_streaming_order_mode: str | None = None,
    full_attention_refine_top_k: int | None = None,
    full_attention_refine_top_k_by_layer: dict[int, int] | None = None,
    full_attention_streaming_priority_value_upper_weight: float | None = None,
    full_attention_streaming_exact_value_rerank_layers: list[int] | None = None,
    full_attention_streaming_exact_value_rerank_max_remaining_blocks: int | None = None,
    full_attention_streaming_refine_top_k: int | None = None,
    full_attention_probe_refine_top_k: int | None = None,
    full_attention_probe_sample_count: int | None = None,
    full_attention_mass_eps: float | None = None,
    full_attention_value_eps: float | None = None,
    full_attention_min_processed_blocks: int | None = None,
    full_attention_region_residual_caps: bool | None = None,
    full_attention_residual_cluster_count: int | None = None,
    full_attention_key_centroid_count: int | None = None,
    full_attention_key_centroid_count_by_layer: dict[int, int] | None = None,
    full_attention_value_centroid_count: int | None = None,
    full_attention_value_centroid_count_by_layer: dict[int, int] | None = None,
    full_attention_streaming_proxy_value_weight_by_layer: dict[int, float] | None = None,
    enable_mixed_execution: bool = False,
    mixed_execution_strategy: str = "cached_reconstruct",
    allow_value_m0: bool = False,
    max_k_comp_error: float | None = 0.10,
) -> PersistentServingConfig:
    config = PersistentServingConfig(
        enable_priority=True,
        enable_early_exit=bool(enable_early_exit),
        enable_compression=bool(enable_mixed_execution),
        enable_full_attention_mixed_mode_execution=bool(enable_mixed_execution),
        full_attention_mixed_mode_execution_strategy=str(mixed_execution_strategy),
        full_attention_mixed_mode_execution_allow_value_m0=bool(allow_value_m0),
        full_attention_mixed_mode_execution_max_k_comp_error=max_k_comp_error,
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
    if full_attention_check_interval is not None:
        config.full_attention_check_interval = max(int(full_attention_check_interval), 1)
    if full_attention_streaming_order_mode is not None:
        config.full_attention_streaming_order_mode = str(full_attention_streaming_order_mode)
    if full_attention_refine_top_k is not None:
        config.full_attention_refine_top_k = max(int(full_attention_refine_top_k), 0)
    if full_attention_refine_top_k_by_layer is not None:
        config.full_attention_refine_top_k_by_layer = {
            int(layer_id): max(int(top_k), 0)
            for layer_id, top_k in dict(full_attention_refine_top_k_by_layer).items()
        }
    if full_attention_streaming_exact_value_rerank_layers is not None:
        config.full_attention_streaming_exact_value_rerank_layers = [
            int(layer_id) for layer_id in list(full_attention_streaming_exact_value_rerank_layers)
        ]
    if full_attention_streaming_priority_value_upper_weight is not None:
        config.full_attention_streaming_priority_value_upper_weight = float(
            full_attention_streaming_priority_value_upper_weight
        )
    if full_attention_streaming_exact_value_rerank_max_remaining_blocks is not None:
        config.full_attention_streaming_exact_value_rerank_max_remaining_blocks = max(
            int(full_attention_streaming_exact_value_rerank_max_remaining_blocks),
            0,
        )
    if full_attention_streaming_refine_top_k is not None:
        config.full_attention_streaming_refine_top_k = max(int(full_attention_streaming_refine_top_k), 0)
    if full_attention_probe_refine_top_k is not None:
        config.full_attention_probe_refine_top_k = max(int(full_attention_probe_refine_top_k), 0)
    if full_attention_probe_sample_count is not None:
        config.full_attention_probe_sample_count = max(int(full_attention_probe_sample_count), 0)
    if full_attention_mass_eps is not None:
        config.full_attention_mass_eps = float(full_attention_mass_eps)
    if full_attention_value_eps is not None:
        config.full_attention_value_eps = float(full_attention_value_eps)
    if full_attention_min_processed_blocks is not None:
        config.full_attention_min_processed_blocks = max(int(full_attention_min_processed_blocks), 1)
    if full_attention_region_residual_caps is not None:
        config.full_attention_region_residual_caps = bool(full_attention_region_residual_caps)
    if full_attention_residual_cluster_count is not None:
        config.full_attention_residual_cluster_count = max(int(full_attention_residual_cluster_count), 0)
    if full_attention_key_centroid_count is not None:
        config.full_attention_key_centroid_count = max(int(full_attention_key_centroid_count), 1)
    if full_attention_key_centroid_count_by_layer is not None:
        config.full_attention_key_centroid_count_by_layer = {
            int(layer_id): max(int(count), 1)
            for layer_id, count in dict(full_attention_key_centroid_count_by_layer).items()
        }
    if full_attention_value_centroid_count is not None:
        config.full_attention_value_centroid_count = max(int(full_attention_value_centroid_count), 1)
    if full_attention_value_centroid_count_by_layer is not None:
        config.full_attention_value_centroid_count_by_layer = {
            int(layer_id): max(int(count), 1)
            for layer_id, count in dict(full_attention_value_centroid_count_by_layer).items()
        }
    if full_attention_streaming_proxy_value_weight_by_layer is not None:
        config.full_attention_streaming_proxy_value_weight_by_layer = {
            int(layer_id): float(weight)
            for layer_id, weight in dict(full_attention_streaming_proxy_value_weight_by_layer).items()
        }
    return config


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
            "hand_tuned_optional_selection_ms_per_case": 0.0,
            "bias_optional_selection_ms_per_case": 0.0,
            "hand_tuned_diverse_selection_ms_per_case": 0.0,
            "bias_diverse_selection_ms_per_case": 0.0,
            "hand_tuned_compression_selection_ms_per_case": 0.0,
            "bias_compression_selection_ms_per_case": 0.0,
            "hand_tuned_policy_bias_ms_per_case": 0.0,
            "bias_policy_bias_ms_per_case": 0.0,
            "hand_tuned_direct_m0_assembly_ms_per_case": 0.0,
            "bias_direct_m0_assembly_ms_per_case": 0.0,
            "hand_tuned_direct_m0_score_ms_per_case": 0.0,
            "bias_direct_m0_score_ms_per_case": 0.0,
            "hand_tuned_exact_m3_score_ms_per_case": 0.0,
            "bias_exact_m3_score_ms_per_case": 0.0,
            "hand_tuned_final_mix_ms_per_case": 0.0,
            "bias_final_mix_ms_per_case": 0.0,
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
        "hand_tuned_optional_selection_ms_per_case": float(
            sum(float(record["hand_tuned_optional_selection_ms_total"]) for record in records) / case_count
        ),
        "bias_optional_selection_ms_per_case": float(
            sum(float(record["bias_optional_selection_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_diverse_selection_ms_per_case": float(
            sum(float(record["hand_tuned_diverse_selection_ms_total"]) for record in records) / case_count
        ),
        "bias_diverse_selection_ms_per_case": float(
            sum(float(record["bias_diverse_selection_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_compression_selection_ms_per_case": float(
            sum(float(record["hand_tuned_compression_selection_ms_total"]) for record in records) / case_count
        ),
        "bias_compression_selection_ms_per_case": float(
            sum(float(record["bias_compression_selection_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_policy_bias_ms_per_case": float(
            sum(float(record["hand_tuned_policy_bias_ms_total"]) for record in records) / case_count
        ),
        "bias_policy_bias_ms_per_case": float(
            sum(float(record["bias_policy_bias_ms_total"]) for record in records) / case_count
        ),
        "hand_tuned_direct_m0_assembly_ms_per_case": float(
            sum(float(record["hand_tuned_direct_m0_assembly_ms_total"]) for record in records) / case_count
        ),
        "bias_direct_m0_assembly_ms_per_case": float(
            sum(float(record["bias_direct_m0_assembly_ms_total"]) for record in records) / case_count
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
    lines.append(
        f"- hand-tuned optional-selection ms/case: {float(summary['hand_tuned_optional_selection_ms_per_case']):.4f}"
    )
    lines.append(f"- bias optional-selection ms/case: {float(summary['bias_optional_selection_ms_per_case']):.4f}")
    lines.append(
        f"- hand-tuned diverse-selection ms/case: {float(summary['hand_tuned_diverse_selection_ms_per_case']):.4f}"
    )
    lines.append(f"- bias diverse-selection ms/case: {float(summary['bias_diverse_selection_ms_per_case']):.4f}")
    lines.append(
        f"- hand-tuned compression-selection ms/case: {float(summary['hand_tuned_compression_selection_ms_per_case']):.4f}"
    )
    lines.append(
        f"- bias compression-selection ms/case: {float(summary['bias_compression_selection_ms_per_case']):.4f}"
    )
    lines.append(f"- hand-tuned policy-bias ms/case: {float(summary['hand_tuned_policy_bias_ms_per_case']):.4f}")
    lines.append(f"- bias policy-bias ms/case: {float(summary['bias_policy_bias_ms_per_case']):.4f}")
    lines.append(
        f"- hand-tuned direct-M0 assembly ms/case: {float(summary['hand_tuned_direct_m0_assembly_ms_per_case']):.4f}"
    )
    lines.append(f"- bias direct-M0 assembly ms/case: {float(summary['bias_direct_m0_assembly_ms_per_case']):.4f}")
    lines.append(f"- hand-tuned direct-M0 score ms/case: {float(summary['hand_tuned_direct_m0_score_ms_per_case']):.4f}")
    lines.append(f"- bias direct-M0 score ms/case: {float(summary['bias_direct_m0_score_ms_per_case']):.4f}")
    lines.append(f"- hand-tuned exact-M3 score ms/case: {float(summary['hand_tuned_exact_m3_score_ms_per_case']):.4f}")
    lines.append(f"- bias exact-M3 score ms/case: {float(summary['bias_exact_m3_score_ms_per_case']):.4f}")
    lines.append(f"- hand-tuned final-mix ms/case: {float(summary['hand_tuned_final_mix_ms_per_case']):.4f}")
    lines.append(f"- bias final-mix ms/case: {float(summary['bias_final_mix_ms_per_case']):.4f}")
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
    full_attention_refine_top_k_by_layer = (
        {
            int(layer_id): max(int(top_k), 0)
            for layer_id, top_k in json.loads(str(args.full_attention_refine_top_k_by_layer)).items()
        }
        if args.full_attention_refine_top_k_by_layer
        else None
    )
    full_attention_key_centroid_count_by_layer = (
        {
            int(layer_id): max(int(count), 1)
            for layer_id, count in json.loads(str(args.full_attention_key_centroid_count_by_layer)).items()
        }
        if args.full_attention_key_centroid_count_by_layer
        else None
    )
    full_attention_streaming_proxy_value_weight_by_layer = (
        {
            int(layer_id): float(weight)
            for layer_id, weight in json.loads(str(args.full_attention_streaming_proxy_value_weight_by_layer)).items()
        }
        if args.full_attention_streaming_proxy_value_weight_by_layer
        else None
    )
    full_attention_streaming_exact_value_rerank_layers = (
        [int(layer_id) for layer_id in json.loads(str(args.full_attention_streaming_exact_value_rerank_layers))]
        if args.full_attention_streaming_exact_value_rerank_layers
        else None
    )
    full_attention_value_centroid_count_by_layer = (
        {
            int(layer_id): max(int(count), 1)
            for layer_id, count in json.loads(str(args.full_attention_value_centroid_count_by_layer)).items()
        }
        if args.full_attention_value_centroid_count_by_layer
        else None
    )
    prompt_records = _resolve_prompt_records(
        manifest_path=args.manifest_path,
        prompt_files=[str(path) for path in args.prompt_files],
        prompt_file_target_length=int(args.prompt_file_target_length),
    )
    if not prompt_records:
        raise SystemExit("no prompt records resolved; provide --manifest-path or --prompt-files")
    allow_value_m0 = bool(args.enable_full_attention_mixed_mode_execution_allow_value_m0)
    mixed_execution_strategy = str(args.full_attention_mixed_mode_execution_strategy)
    max_k_comp_error = _parse_optional_float(args.full_attention_mixed_mode_execution_max_k_comp_error)

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
        persistent_serving_config=_persistent_base_config(
            policy_path=None,
            enable_early_exit=bool(args.enable_early_exit),
            full_attention_check_interval=args.full_attention_check_interval,
            full_attention_streaming_order_mode=args.full_attention_streaming_order_mode,
            full_attention_refine_top_k=args.full_attention_refine_top_k,
            full_attention_refine_top_k_by_layer=full_attention_refine_top_k_by_layer,
            full_attention_streaming_priority_value_upper_weight=(
                args.full_attention_streaming_priority_value_upper_weight
            ),
            full_attention_streaming_exact_value_rerank_layers=full_attention_streaming_exact_value_rerank_layers,
            full_attention_streaming_exact_value_rerank_max_remaining_blocks=(
                args.full_attention_streaming_exact_value_rerank_max_remaining_blocks
            ),
            full_attention_streaming_refine_top_k=args.full_attention_streaming_refine_top_k,
            full_attention_probe_refine_top_k=args.full_attention_probe_refine_top_k,
            full_attention_probe_sample_count=args.full_attention_probe_sample_count,
            full_attention_mass_eps=args.full_attention_mass_eps,
            full_attention_value_eps=args.full_attention_value_eps,
            full_attention_min_processed_blocks=args.full_attention_min_processed_blocks,
            full_attention_region_residual_caps=args.full_attention_region_residual_caps,
            full_attention_residual_cluster_count=args.full_attention_residual_cluster_count,
            full_attention_key_centroid_count=args.full_attention_key_centroid_count,
            full_attention_key_centroid_count_by_layer=full_attention_key_centroid_count_by_layer,
            full_attention_value_centroid_count=args.full_attention_value_centroid_count,
            full_attention_value_centroid_count_by_layer=full_attention_value_centroid_count_by_layer,
            full_attention_streaming_proxy_value_weight_by_layer=full_attention_streaming_proxy_value_weight_by_layer,
            enable_mixed_execution=bool(args.enable_full_attention_mixed_mode_execution),
            mixed_execution_strategy=mixed_execution_strategy,
            allow_value_m0=allow_value_m0,
            max_k_comp_error=max_k_comp_error,
        ),
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
        persistent_adapter.persistent_serving_config = _persistent_base_config(
            policy_path=None,
            enable_early_exit=bool(args.enable_early_exit),
            full_attention_check_interval=args.full_attention_check_interval,
            full_attention_streaming_order_mode=args.full_attention_streaming_order_mode,
            full_attention_refine_top_k=args.full_attention_refine_top_k,
            full_attention_refine_top_k_by_layer=full_attention_refine_top_k_by_layer,
            full_attention_streaming_priority_value_upper_weight=(
                args.full_attention_streaming_priority_value_upper_weight
            ),
            full_attention_streaming_exact_value_rerank_layers=full_attention_streaming_exact_value_rerank_layers,
            full_attention_streaming_exact_value_rerank_max_remaining_blocks=(
                args.full_attention_streaming_exact_value_rerank_max_remaining_blocks
            ),
            full_attention_streaming_refine_top_k=args.full_attention_streaming_refine_top_k,
            full_attention_probe_refine_top_k=args.full_attention_probe_refine_top_k,
            full_attention_probe_sample_count=args.full_attention_probe_sample_count,
            full_attention_mass_eps=args.full_attention_mass_eps,
            full_attention_value_eps=args.full_attention_value_eps,
            full_attention_min_processed_blocks=args.full_attention_min_processed_blocks,
            full_attention_region_residual_caps=args.full_attention_region_residual_caps,
            full_attention_residual_cluster_count=args.full_attention_residual_cluster_count,
            full_attention_key_centroid_count=args.full_attention_key_centroid_count,
            full_attention_key_centroid_count_by_layer=full_attention_key_centroid_count_by_layer,
            full_attention_value_centroid_count=args.full_attention_value_centroid_count,
            full_attention_value_centroid_count_by_layer=full_attention_value_centroid_count_by_layer,
            full_attention_streaming_proxy_value_weight_by_layer=full_attention_streaming_proxy_value_weight_by_layer,
            enable_mixed_execution=bool(args.enable_full_attention_mixed_mode_execution),
            mixed_execution_strategy=mixed_execution_strategy,
            allow_value_m0=allow_value_m0,
            max_k_comp_error=max_k_comp_error,
        )
        hand_result = run_qwen35_attention_subset_persistent_serving_harness(
            persistent_model,
            persistent_adapter,
            input_ids=persistent_input_ids,
            attention_mask=persistent_attention_mask,
            tokenizer=persistent_tokenizer,
            decode_steps=int(args.decode_steps),
            persistent_policy_prompt_family=str(prompt_record["case_tag"]),
        )
        persistent_adapter.persistent_serving_config = _persistent_base_config(
            policy_path=str(args.shortlist_policy_path),
            enable_early_exit=bool(args.enable_early_exit),
            full_attention_check_interval=args.full_attention_check_interval,
            full_attention_streaming_order_mode=args.full_attention_streaming_order_mode,
            full_attention_refine_top_k=args.full_attention_refine_top_k,
            full_attention_refine_top_k_by_layer=full_attention_refine_top_k_by_layer,
            full_attention_streaming_priority_value_upper_weight=(
                args.full_attention_streaming_priority_value_upper_weight
            ),
            full_attention_streaming_exact_value_rerank_layers=full_attention_streaming_exact_value_rerank_layers,
            full_attention_streaming_exact_value_rerank_max_remaining_blocks=(
                args.full_attention_streaming_exact_value_rerank_max_remaining_blocks
            ),
            full_attention_streaming_refine_top_k=args.full_attention_streaming_refine_top_k,
            full_attention_probe_refine_top_k=args.full_attention_probe_refine_top_k,
            full_attention_probe_sample_count=args.full_attention_probe_sample_count,
            full_attention_mass_eps=args.full_attention_mass_eps,
            full_attention_value_eps=args.full_attention_value_eps,
            full_attention_min_processed_blocks=args.full_attention_min_processed_blocks,
            full_attention_region_residual_caps=args.full_attention_region_residual_caps,
            full_attention_residual_cluster_count=args.full_attention_residual_cluster_count,
            full_attention_key_centroid_count=args.full_attention_key_centroid_count,
            full_attention_key_centroid_count_by_layer=full_attention_key_centroid_count_by_layer,
            full_attention_value_centroid_count=args.full_attention_value_centroid_count,
            full_attention_value_centroid_count_by_layer=full_attention_value_centroid_count_by_layer,
            full_attention_streaming_proxy_value_weight_by_layer=full_attention_streaming_proxy_value_weight_by_layer,
            enable_mixed_execution=bool(args.enable_full_attention_mixed_mode_execution),
            mixed_execution_strategy=mixed_execution_strategy,
            allow_value_m0=allow_value_m0,
            max_k_comp_error=max_k_comp_error,
        )
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
            "hand_tuned_last_processed_block_count": hand_result.get(
                "persistent_full_attention_last_processed_block_count_by_layer",
                {},
            ).get("3"),
            "bias_last_processed_block_count": bias_result.get(
                "persistent_full_attention_last_processed_block_count_by_layer",
                {},
            ).get("3"),
            "hand_tuned_last_certified_can_stop": hand_result.get(
                "persistent_full_attention_last_certified_can_stop_by_layer",
                {},
            ).get("3"),
            "bias_last_certified_can_stop": bias_result.get(
                "persistent_full_attention_last_certified_can_stop_by_layer",
                {},
            ).get("3"),
            "hand_tuned_first_certified_stop_block_count": hand_result.get(
                "persistent_full_attention_last_first_certified_stop_block_count_by_layer",
                {},
            ).get("3"),
            "bias_first_certified_stop_block_count": bias_result.get(
                "persistent_full_attention_last_first_certified_stop_block_count_by_layer",
                {},
            ).get("3"),
            "hand_tuned_last_checkpoint_count": hand_result.get(
                "persistent_full_attention_last_checkpoint_count_by_layer",
                {},
            ).get("3"),
            "bias_last_checkpoint_count": bias_result.get(
                "persistent_full_attention_last_checkpoint_count_by_layer",
                {},
            ).get("3"),
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
            "hand_tuned_optional_selection_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get("persistent_full_attention_optional_selection_ms_total_by_layer", {}).values()
                )
            ),
            "bias_optional_selection_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get("persistent_full_attention_optional_selection_ms_total_by_layer", {}).values()
                )
            ),
            "hand_tuned_diverse_selection_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get("persistent_full_attention_diverse_selection_ms_total_by_layer", {}).values()
                )
            ),
            "bias_diverse_selection_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get("persistent_full_attention_diverse_selection_ms_total_by_layer", {}).values()
                )
            ),
            "hand_tuned_compression_selection_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get("persistent_full_attention_compression_selection_ms_total_by_layer", {}).values()
                )
            ),
            "bias_compression_selection_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get("persistent_full_attention_compression_selection_ms_total_by_layer", {}).values()
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
            "hand_tuned_direct_m0_assembly_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get(
                        "persistent_full_attention_mixed_execution_direct_m0_assembly_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "bias_direct_m0_assembly_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get(
                        "persistent_full_attention_mixed_execution_direct_m0_assembly_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "hand_tuned_direct_m0_score_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get(
                        "persistent_full_attention_mixed_execution_direct_m0_score_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "bias_direct_m0_score_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get(
                        "persistent_full_attention_mixed_execution_direct_m0_score_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "hand_tuned_exact_m3_score_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get(
                        "persistent_full_attention_mixed_execution_exact_m3_score_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "bias_exact_m3_score_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get(
                        "persistent_full_attention_mixed_execution_exact_m3_score_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "hand_tuned_final_mix_ms_total": float(
                sum(
                    float(value)
                    for value in hand_result.get(
                        "persistent_full_attention_mixed_execution_final_mix_ms_total_by_layer",
                        {},
                    ).values()
                )
            ),
            "bias_final_mix_ms_total": float(
                sum(
                    float(value)
                    for value in bias_result.get(
                        "persistent_full_attention_mixed_execution_final_mix_ms_total_by_layer",
                        {},
                    ).values()
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
            "enable_early_exit": bool(args.enable_early_exit),
            "full_attention_check_interval": args.full_attention_check_interval,
            "full_attention_refine_top_k": args.full_attention_refine_top_k,
            "full_attention_refine_top_k_by_layer": full_attention_refine_top_k_by_layer,
            "full_attention_streaming_exact_value_rerank_layers": (
                full_attention_streaming_exact_value_rerank_layers
            ),
            "full_attention_streaming_priority_value_upper_weight": (
                args.full_attention_streaming_priority_value_upper_weight
            ),
            "full_attention_streaming_exact_value_rerank_max_remaining_blocks": (
                args.full_attention_streaming_exact_value_rerank_max_remaining_blocks
            ),
            "full_attention_mass_eps": args.full_attention_mass_eps,
            "full_attention_value_eps": args.full_attention_value_eps,
            "full_attention_min_processed_blocks": args.full_attention_min_processed_blocks,
            "full_attention_probe_refine_top_k": args.full_attention_probe_refine_top_k,
            "full_attention_probe_sample_count": args.full_attention_probe_sample_count,
            "full_attention_region_residual_caps": args.full_attention_region_residual_caps,
            "full_attention_residual_cluster_count": args.full_attention_residual_cluster_count,
            "full_attention_key_centroid_count": args.full_attention_key_centroid_count,
            "full_attention_key_centroid_count_by_layer": full_attention_key_centroid_count_by_layer,
            "full_attention_value_centroid_count": args.full_attention_value_centroid_count,
            "full_attention_value_centroid_count_by_layer": full_attention_value_centroid_count_by_layer,
            "full_attention_streaming_proxy_value_weight_by_layer": (
                full_attention_streaming_proxy_value_weight_by_layer
            ),
            "enable_full_attention_mixed_mode_execution": bool(args.enable_full_attention_mixed_mode_execution),
            "full_attention_mixed_mode_execution_strategy": mixed_execution_strategy,
            "enable_full_attention_mixed_mode_execution_allow_value_m0": bool(allow_value_m0),
            "full_attention_mixed_mode_execution_max_k_comp_error": max_k_comp_error,
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
