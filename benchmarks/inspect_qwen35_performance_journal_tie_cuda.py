from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_real_mixed_probe import (  # noqa: E402
    _REAL_MIXED_KEY_CENTROIDS,
    real_mixed_probe_dotcache_config,
)
from benchmarks.bench_qwen35_persistent_serving_policy_compare import (  # noqa: E402
    _build_prompt_text_inputs,
    _persistent_base_config,
)
from dotcache.backends.metal import persistent_runtime as persistent_runtime_mod  # noqa: E402
from dotcache.integrations import qwen35 as qwen35_mod  # noqa: E402
from dotcache.integrations.qwen35 import (  # noqa: E402
    Qwen35AttentionSubsetDotCacheModelAdapter,
    _prepare_qwen35_attention_subset_dotcache_runtime,
    _run_dense_decode_step,
    _run_qwen35_attention_subset_dense_capture,
    load_qwen35_text_only_from_pretrained,
)


_DEFAULT_PROMPT_PATH = Path("docs/performance_journal.md")
_DEFAULT_OUTPUT_DIR = Path(
    "benchmarks/results/qwen35_performance_journal_tie_cuda_diag_20260413"
)


class _DiagState:
    def __init__(self, *, target_step: int) -> None:
        self.target_step = int(target_step)
        self.active_lane: str | None = None
        self.active_step: int | None = None
        self.active_layer: int | None = None
        self.final_mix_entries: list[dict[str, Any]] = []
        self.layer_decode_entries: list[dict[str, Any]] = []
        self.force_stream_attn_capture: bool = False

    def clear_lane(self, lane_name: str) -> None:
        self.active_lane = str(lane_name)
        self.active_step = None
        self.active_layer = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the CUDA performance_journal tie-boundary step.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["torch_cuda", "auto"], default="torch_cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--target-step", type=int, default=5)
    parser.add_argument("--prompt-path", default=str(_DEFAULT_PROMPT_PATH))
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--force-stream-attn-capture", action="store_true")
    parser.add_argument("--output-json", default=str(_DEFAULT_OUTPUT_DIR / "diagnostic.json"))
    parser.add_argument("--output-md", default=str(_DEFAULT_OUTPUT_DIR / "diagnostic.md"))
    return parser.parse_args()


def _real_mixed_config() -> Any:
    return _persistent_base_config(
        policy_path=None,
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


def _non_m0_config() -> Any:
    return _persistent_base_config(
        policy_path=None,
        enable_early_exit=True,
        full_attention_check_interval=16,
        full_attention_streaming_order_mode="priority_value_hybrid",
        full_attention_streaming_priority_value_upper_weight=0.25,
        full_attention_key_centroid_count_by_layer=dict(_REAL_MIXED_KEY_CENTROIDS),
        enable_mixed_execution=False,
        mixed_execution_strategy="cached_reconstruct",
        allow_value_m0=False,
        max_k_comp_error=0.20,
    )


def _topk_summary(tensor_2d: torch.Tensor, *, k: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values, indices = torch.topk(tensor_2d, k=min(int(k), int(tensor_2d.shape[-1])), dim=-1)
    for row_idx in range(int(tensor_2d.shape[0])):
        rows.append(
            {
                "row_index": int(row_idx),
                "indices": [int(value) for value in indices[row_idx].detach().cpu().tolist()],
                "values": [float(value) for value in values[row_idx].detach().to(dtype=torch.float32).cpu().tolist()],
            }
        )
    return rows


def _resolve_block_ids(block_ids: Any) -> list[int]:
    if torch.is_tensor(block_ids):
        return [int(value) for value in block_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()]
    return [int(value) for value in block_ids]


def _gather_selected_values_for_block_ids(state: Any, block_ids: list[int]) -> torch.Tensor:
    token_indices: list[int] = []
    for block_id in block_ids:
        token_start = int(state.block_token_starts[int(block_id)])
        token_count = int(state.block_token_counts[int(block_id)])
        token_indices.extend(range(token_start, token_start + token_count))
    index_tensor = torch.as_tensor(token_indices, dtype=torch.int64, device=state.value_cache.device)
    return state.value_cache.index_select(1, index_tensor).to(dtype=torch.float32)


@contextmanager
def _install_tie_step_hooks(diag_state: _DiagState):
    original_decode = persistent_runtime_mod.PersistentHybridRuntimeState.decode_full_attention_selected_blocks
    original_stream = persistent_runtime_mod.PersistentHybridRuntimeState.stream_full_attention_layer
    original_direct_m0 = persistent_runtime_mod._decode_selected_blocks_direct_m0_torch
    original_fast = persistent_runtime_mod._cuda_direct_m0_fast_final_mix_exact_values
    original_triton = persistent_runtime_mod._cuda_direct_m0_triton_softmax_final_mix_exact_values
    original_native = persistent_runtime_mod._cuda_direct_m0_native_final_mix_exact_values

    def _capture_final_mix(
        *,
        path_name: str,
        logits: Any,
        gathered_values: Any,
        query_scale: float,
        call_original,
        **kwargs: Any,
    ):
        logits_f32 = logits.detach().to(dtype=torch.float32)
        gathered_values_f32 = gathered_values.detach().to(dtype=torch.float32)
        ref_scaled_logits = logits_f32 * float(query_scale)
        ref_weights = torch.softmax(ref_scaled_logits, dim=-1)
        ref_context = torch.matmul(ref_weights, gathered_values_f32).to(dtype=torch.float32)
        original_result = call_original(
            logits=logits,
            gathered_values=gathered_values,
            query_scale=query_scale,
            **kwargs,
        )
        if path_name == "native":
            actual_context = original_result.detach().to(dtype=torch.float32)
            actual_weights = None
        else:
            actual_context, actual_weights = original_result
            actual_context = actual_context.detach().to(dtype=torch.float32)
            actual_weights = actual_weights.detach().to(dtype=torch.float32)
        entry = {
            "lane": diag_state.active_lane,
            "step_index": diag_state.active_step,
            "layer_id": diag_state.active_layer,
            "path": path_name,
            "query_scale": float(query_scale),
            "logits_shape": list(logits_f32.shape),
            "gathered_values_shape": list(gathered_values_f32.shape),
            "logits_topk_by_row": _topk_summary(ref_scaled_logits, k=4),
            "weights_topk_by_row": _topk_summary(ref_weights, k=4),
            "weights_row_sum": [
                float(value) for value in ref_weights.sum(dim=-1).detach().to(dtype=torch.float32).cpu().tolist()
            ],
            "gathered_values_rms": float(torch.sqrt(torch.mean(torch.square(gathered_values_f32))).item()),
            "gathered_values_max_abs": float(torch.max(torch.abs(gathered_values_f32)).item()),
            "context_rms": float(torch.sqrt(torch.mean(torch.square(actual_context))).item()),
            "context_max_abs": float(torch.max(torch.abs(actual_context)).item()),
            "context_ref_max_abs_delta": float(torch.max(torch.abs(actual_context - ref_context)).item()),
            "context_ref_mean_abs_delta": float(torch.mean(torch.abs(actual_context - ref_context)).item()),
        }
        if actual_weights is not None:
            entry["weights_ref_max_abs_delta"] = float(torch.max(torch.abs(actual_weights - ref_weights)).item())
            entry["weights_ref_mean_abs_delta"] = float(torch.mean(torch.abs(actual_weights - ref_weights)).item())
        diag_state.final_mix_entries.append(entry)
        return original_result

    def _wrapped_direct_m0(
        *,
        state: Any,
        block_ids: Any,
        query: Any,
        q_head_to_kv_head: np.ndarray,
        query_scale: float,
        config: Any,
        dotcache_config: Any | None = None,
        return_stream_stats: bool = False,
        return_attn_weights: bool = True,
    ):
        result = original_direct_m0(
            state=state,
            block_ids=block_ids,
            query=query,
            q_head_to_kv_head=q_head_to_kv_head,
            query_scale=query_scale,
            config=config,
            dotcache_config=dotcache_config,
            return_stream_stats=return_stream_stats,
            return_attn_weights=return_attn_weights,
        )
        if diag_state.active_step != diag_state.target_step or not bool(return_stream_stats):
            return result
        resolved_block_ids = _resolve_block_ids(block_ids)
        gathered_values = _gather_selected_values_for_block_ids(state, resolved_block_ids)
        q_to_kv = np.asarray(q_head_to_kv_head, dtype=np.int64)
        per_head_logits = result["stream_stats"]["per_head_logits"]
        if per_head_logits is None:
            diag_state.final_mix_entries.append(
                {
                    "lane": diag_state.active_lane,
                    "step_index": diag_state.active_step,
                    "layer_id": diag_state.active_layer,
                    "path": "direct_m0_stream_stats",
                    "selected_block_ids": [int(value) for value in resolved_block_ids],
                    "selected_block_count": int(len(resolved_block_ids)),
                    "token_count": int(gathered_values.shape[1]),
                    "executed_mode_counts": {
                        key: int(value) for key, value in result.get("executed_mode_counts", {}).items()
                    },
                    "gathered_values_rms": float(torch.sqrt(torch.mean(torch.square(gathered_values))).item()),
                    "gathered_values_max_abs": float(torch.max(torch.abs(gathered_values)).item()),
                    "context_ref_max_abs_delta": None,
                    "context_ref_mean_abs_delta": None,
                    "per_head": [],
                }
            )
            return result
        output = result["output"].detach().to(dtype=torch.float32)
        ref_context = torch.zeros_like(output)
        per_head_entries: list[dict[str, Any]] = []
        for q_head_idx, head_logits in enumerate(per_head_logits):
            if head_logits is None or int(head_logits.numel()) <= 0:
                continue
            logits_f32 = head_logits.detach().to(dtype=torch.float32)
            weights = torch.softmax(logits_f32, dim=-1)
            kv_head_idx = int(q_to_kv[int(q_head_idx)])
            head_context = torch.matmul(weights, gathered_values[kv_head_idx]).to(dtype=torch.float32)
            ref_context[int(q_head_idx)] = head_context
            top_logit_values, top_logit_indices = torch.topk(logits_f32, k=min(4, int(logits_f32.numel())))
            top_weight_values, _ = torch.topk(weights, k=min(4, int(weights.numel())))
            per_head_entries.append(
                {
                    "q_head_id": int(q_head_idx),
                    "kv_head_id": int(kv_head_idx),
                    "top_token_indices": [int(value) for value in top_logit_indices.detach().cpu().tolist()],
                    "top_logits": [float(value) for value in top_logit_values.detach().cpu().tolist()],
                    "weight_sum": float(weights.sum().item()),
                    "top_weight_values": [float(value) for value in top_weight_values.detach().cpu().tolist()],
                }
            )
        context_delta = torch.abs(output - ref_context)
        diag_state.final_mix_entries.append(
            {
                "lane": diag_state.active_lane,
                "step_index": diag_state.active_step,
                "layer_id": diag_state.active_layer,
                "path": "direct_m0_stream_stats",
                "query_scale": float(query_scale),
                "selected_block_ids": [int(value) for value in resolved_block_ids],
                "selected_block_count": int(len(resolved_block_ids)),
                "token_count": int(gathered_values.shape[1]),
                "executed_mode_counts": {
                    key: int(value) for key, value in result.get("executed_mode_counts", {}).items()
                },
                "gathered_values_rms": float(torch.sqrt(torch.mean(torch.square(gathered_values))).item()),
                "gathered_values_max_abs": float(torch.max(torch.abs(gathered_values)).item()),
                "context_ref_max_abs_delta": float(torch.max(context_delta).item()),
                "context_ref_mean_abs_delta": float(torch.mean(context_delta).item()),
                "per_head": per_head_entries,
            }
        )
        return result

    def _wrapped_fast(*, logits: Any, gathered_values: Any, query_scale: float, score_dtype: Any):
        if diag_state.active_step != diag_state.target_step:
            return original_fast(
                logits=logits,
                gathered_values=gathered_values,
                query_scale=query_scale,
                score_dtype=score_dtype,
            )
        return _capture_final_mix(
            path_name="fast",
            logits=logits,
            gathered_values=gathered_values,
            query_scale=query_scale,
            call_original=original_fast,
            score_dtype=score_dtype,
        )

    def _wrapped_triton(
        *,
        logits: Any,
        gathered_values: Any,
        query_scale: float,
        score_dtype: Any,
        softmax_weights_triton: Any,
    ):
        if diag_state.active_step != diag_state.target_step:
            return original_triton(
                logits=logits,
                gathered_values=gathered_values,
                query_scale=query_scale,
                score_dtype=score_dtype,
                softmax_weights_triton=softmax_weights_triton,
            )
        return _capture_final_mix(
            path_name="triton",
            logits=logits,
            gathered_values=gathered_values,
            query_scale=query_scale,
            call_original=original_triton,
            score_dtype=score_dtype,
            softmax_weights_triton=softmax_weights_triton,
        )

    def _wrapped_native(
        *,
        logits: Any,
        gathered_values: Any,
        query_scale: float,
        score_dtype: Any,
        softmax_value_context_cuda: Any,
    ):
        if diag_state.active_step != diag_state.target_step:
            return original_native(
                logits=logits,
                gathered_values=gathered_values,
                query_scale=query_scale,
                score_dtype=score_dtype,
                softmax_value_context_cuda=softmax_value_context_cuda,
            )
        return _capture_final_mix(
            path_name="native",
            logits=logits,
            gathered_values=gathered_values,
            query_scale=query_scale,
            call_original=original_native,
            score_dtype=score_dtype,
            softmax_value_context_cuda=softmax_value_context_cuda,
        )

    def _wrapped_decode(self, layer_id: int, *args: Any, **kwargs: Any):
        previous_layer = diag_state.active_layer
        diag_state.active_layer = int(layer_id)
        try:
            result = original_decode(self, layer_id, *args, **kwargs)
        finally:
            diag_state.active_layer = previous_layer
        if diag_state.active_step == diag_state.target_step:
            diag_state.layer_decode_entries.append(
                {
                    "lane": diag_state.active_lane,
                    "step_index": diag_state.active_step,
                    "layer_id": int(layer_id),
                    "processed_block_count": int(result.get("processed_block_count", 0)),
                    "processed_token_count": int(result.get("processed_token_count", 0)),
                    "checkpoint_count": int(len(result.get("checkpoint_records", []))),
                    "first_certified_stop": copy.deepcopy(result.get("first_certified_stop")),
                    "final_checkpoint": copy.deepcopy(result.get("final_checkpoint")),
                    "selection_block_ids": [
                        int(value) for value in result.get("selection", {}).get("selected_block_ids", [])
                    ],
                }
            )
        return result

    def _wrapped_stream(self, layer_id: int, query: Any, **kwargs: Any):
        previous_layer = diag_state.active_layer
        diag_state.active_layer = int(layer_id)
        if (
            diag_state.force_stream_attn_capture
            and diag_state.active_step == diag_state.target_step
            and diag_state.active_lane == "real_mixed"
        ):
            kwargs = dict(kwargs)
            kwargs["return_attn_weights"] = True
            kwargs["return_checkpoint_records"] = True
            kwargs["return_checkpoint_per_head"] = True
            kwargs["return_certificate_summary_only"] = False
        try:
            result = original_stream(self, layer_id, query, **kwargs)
        finally:
            diag_state.active_layer = previous_layer
        if diag_state.active_step == diag_state.target_step:
            diag_state.layer_decode_entries.append(
                {
                    "lane": diag_state.active_lane,
                    "step_index": diag_state.active_step,
                    "layer_id": int(layer_id),
                    "processed_block_count": int(result.get("processed_block_count", 0)),
                    "processed_token_count": int(result.get("processed_token_count", 0)),
                    "checkpoint_count": int(len(result.get("checkpoint_records", []))),
                    "first_certified_stop": copy.deepcopy(result.get("first_certified_stop")),
                    "final_checkpoint": copy.deepcopy(result.get("final_checkpoint")),
                    "selection_block_ids": [
                        int(value) for value in result.get("selection", {}).get("selected_block_ids", [])
                    ],
                }
            )
        return result

    persistent_runtime_mod.PersistentHybridRuntimeState.decode_full_attention_selected_blocks = _wrapped_decode
    persistent_runtime_mod.PersistentHybridRuntimeState.stream_full_attention_layer = _wrapped_stream
    persistent_runtime_mod._decode_selected_blocks_direct_m0_torch = _wrapped_direct_m0
    persistent_runtime_mod._cuda_direct_m0_fast_final_mix_exact_values = _wrapped_fast
    persistent_runtime_mod._cuda_direct_m0_triton_softmax_final_mix_exact_values = _wrapped_triton
    persistent_runtime_mod._cuda_direct_m0_native_final_mix_exact_values = _wrapped_native
    try:
        yield
    finally:
        persistent_runtime_mod.PersistentHybridRuntimeState.decode_full_attention_selected_blocks = original_decode
        persistent_runtime_mod.PersistentHybridRuntimeState.stream_full_attention_layer = original_stream
        persistent_runtime_mod._decode_selected_blocks_direct_m0_torch = original_direct_m0
        persistent_runtime_mod._cuda_direct_m0_fast_final_mix_exact_values = original_fast
        persistent_runtime_mod._cuda_direct_m0_triton_softmax_final_mix_exact_values = original_triton
        persistent_runtime_mod._cuda_direct_m0_native_final_mix_exact_values = original_native


def _run_persistent_capture(
    *,
    model: Any,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
    lane_name: str,
    persistent_serving_config: Any,
    diag_state: _DiagState,
) -> dict[str, Any]:
    adapter.persistent_serving_config = persistent_serving_config
    prepared = _prepare_qwen35_attention_subset_dotcache_runtime(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        runtime_mode="dotcache_attention_subset_persistent_experimental",
    )
    runtime_state = prepared["runtime_state"]
    prefill_outputs = prepared["dotcache_prefill_outputs"]
    current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    current_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    generated_ids: list[int] = []
    step_logits: list[np.ndarray] = []
    per_step_records: list[list[Any]] = []
    for step_index in range(int(decode_steps)):
        diag_state.active_step = int(step_index)
        generated_ids.append(int(current_input_ids.item()))
        adapter.begin_capture_step(step_index)
        adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
        try:
            outputs = _run_dense_decode_step(
                model,
                decode_input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=runtime_state.model_past_key_values,
                cache_position=cache_position,
            )
        finally:
            adapter.set_current_token_index(None)
        per_step_records.append(adapter.end_capture_step())
        step_logits.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu().numpy())
        runtime_state.advance(outputs.past_key_values)
        current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1
    diag_state.active_step = None
    return {
        "lane": str(lane_name),
        "generated_ids": [int(value) for value in generated_ids],
        "step_logits": step_logits,
        "capture_records": per_step_records,
    }


def _token_readout(step_logits: np.ndarray, *, token_ids: list[int]) -> dict[str, Any]:
    logits = np.asarray(step_logits, dtype=np.float32).reshape(-1)
    stabilized = logits - float(np.max(logits))
    probs = np.exp(stabilized)
    probs = probs / max(float(np.sum(probs)), 1e-20)
    top_indices = np.argsort(logits)[-8:][::-1]
    return {
        "argmax_token_id": int(np.argmax(logits)),
        "token_logits": {str(token_id): float(logits[int(token_id)]) for token_id in token_ids},
        "token_probs": {str(token_id): float(probs[int(token_id)]) for token_id in token_ids},
        "token_logit_margin_15_minus_16": float(logits[15] - logits[16]),
        "top_tokens": [
            {
                "token_id": int(token_id),
                "logit": float(logits[int(token_id)]),
                "prob": float(probs[int(token_id)]),
            }
            for token_id in top_indices
        ],
    }


def _layer_record_map(step_records: list[Any]) -> dict[int, Any]:
    return {int(record.layer_id): record for record in step_records}


def _per_layer_deltas(reference_records: list[Any], probe_records: list[Any]) -> list[dict[str, Any]]:
    ref_map = _layer_record_map(reference_records)
    probe_map = _layer_record_map(probe_records)
    rows: list[dict[str, Any]] = []
    for layer_id in sorted(set(ref_map) | set(probe_map)):
        ref_record = ref_map.get(layer_id)
        probe_record = probe_map.get(layer_id)
        if ref_record is None or probe_record is None:
            rows.append(
                {
                    "layer_id": int(layer_id),
                    "missing_in_reference": bool(ref_record is None),
                    "missing_in_probe": bool(probe_record is None),
                }
            )
            continue
        context_delta = np.abs(
            np.asarray(probe_record.context_states, dtype=np.float32)
            - np.asarray(ref_record.context_states, dtype=np.float32)
        )
        output_delta = np.abs(
            np.asarray(probe_record.output_states, dtype=np.float32)
            - np.asarray(ref_record.output_states, dtype=np.float32)
        )
        gate_delta = None
        if getattr(ref_record, "gate_states", None) is not None and getattr(probe_record, "gate_states", None) is not None:
            gate_delta = np.abs(
                np.asarray(probe_record.gate_states, dtype=np.float32)
                - np.asarray(ref_record.gate_states, dtype=np.float32)
            )
        rows.append(
            {
                "layer_id": int(layer_id),
                "context_max_abs": float(np.max(context_delta)),
                "context_mean_abs": float(np.mean(context_delta)),
                "output_max_abs": float(np.max(output_delta)),
                "output_mean_abs": float(np.mean(output_delta)),
                "gate_max_abs": (None if gate_delta is None else float(np.max(gate_delta))),
                "gate_mean_abs": (None if gate_delta is None else float(np.mean(gate_delta))),
            }
        )
    return rows


def _first_nonzero_output_delta(layer_deltas: list[dict[str, Any]], *, eps: float = 1e-6) -> int | None:
    for row in layer_deltas:
        if float(row.get("output_max_abs", 0.0) or 0.0) > float(eps):
            return int(row["layer_id"])
    return None


def _render_markdown(payload: dict[str, Any]) -> str:
    target_step = int(payload["target_step"])
    dense = payload["lanes"]["dense"]
    real_mixed = payload["lanes"]["real_mixed"]
    non_m0 = payload["lanes"]["non_m0"]
    comparison = payload["comparisons"]["real_mixed_vs_non_m0"]
    lines = [
        "# Qwen3.5 `performance_journal` CUDA Tie-Step Diagnostic",
        "",
        "## Summary",
        "",
        f"- target decode step: `{target_step}`",
        f"- dense generated ids: `{dense['generated_ids']}`",
        f"- real mixed generated ids: `{real_mixed['generated_ids']}`",
        f"- non-M0 generated ids: `{non_m0['generated_ids']}`",
        f"- dense step-{target_step} argmax: `{dense['target_step_readout']['argmax_token_id']}`",
        f"- real mixed step-{target_step} argmax: `{real_mixed['target_step_readout']['argmax_token_id']}`",
        f"- non-M0 step-{target_step} argmax: `{non_m0['target_step_readout']['argmax_token_id']}`",
        f"- first real-mixed vs non-M0 output-layer delta above `1e-6`: `{comparison['first_nonzero_output_delta_layer_id']}`",
        "",
        "## Token 15/16 Readout",
        "",
        f"- dense logits: `{dense['target_step_readout']['token_logits']}`",
        f"- dense probs: `{dense['target_step_readout']['token_probs']}`",
        f"- real mixed logits: `{real_mixed['target_step_readout']['token_logits']}`",
        f"- real mixed probs: `{real_mixed['target_step_readout']['token_probs']}`",
        f"- non-M0 logits: `{non_m0['target_step_readout']['token_logits']}`",
        f"- non-M0 probs: `{non_m0['target_step_readout']['token_probs']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "## Final-Mix Entries",
        "",
    ]
    for entry in payload["real_mixed_final_mix"]:
        context_delta = entry.get("context_ref_max_abs_delta")
        weights_delta = entry.get("weights_ref_max_abs_delta")
        lines.append(
            f"- layer `{entry['layer_id']}` `{entry['path']}` context-ref max abs delta: "
            f"`{'n/a' if context_delta is None else f'{float(context_delta):.8f}'}`, weights-ref max abs delta: "
            f"`{'n/a' if weights_delta is None else f'{float(weights_delta):.8f}'}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_path).resolve()
    prompt_text = prompt_path.read_text(encoding="utf-8")

    model, tokenizer = load_qwen35_text_only_from_pretrained(
        str(args.model_id),
        device=str(args.device),
        torch_dtype=str(args.torch_dtype),
        weight_quantization="none",
    )
    input_ids, attention_mask = _build_prompt_text_inputs(
        tokenizer,
        device=next(model.parameters()).device,
        prompt_text=prompt_text,
        prompt_length=int(args.prompt_length),
    )
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=real_mixed_probe_dotcache_config(),
        persistent_serving_config=_real_mixed_config(),
        backend=str(args.backend),
    )

    dense_capture = _run_qwen35_attention_subset_dense_capture(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=int(args.decode_steps),
    )
    dense_generated_ids = []
    for decode_input_ids in dense_capture["decode_inputs"]:
        if torch.is_tensor(decode_input_ids):
            dense_generated_ids.append(int(decode_input_ids.detach().view(-1)[0].item()))
        else:
            dense_generated_ids.append(int(np.asarray(decode_input_ids).reshape(-1)[0]))
    diag_state = _DiagState(target_step=int(args.target_step))
    diag_state.force_stream_attn_capture = bool(args.force_stream_attn_capture)
    with _install_tie_step_hooks(diag_state):
        diag_state.clear_lane("real_mixed")
        real_mixed_capture = _run_persistent_capture(
            model=model,
            adapter=adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=int(args.decode_steps),
            lane_name="real_mixed",
            persistent_serving_config=_real_mixed_config(),
            diag_state=diag_state,
        )
        diag_state.clear_lane("non_m0")
        non_m0_capture = _run_persistent_capture(
            model=model,
            adapter=adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=int(args.decode_steps),
            lane_name="non_m0",
            persistent_serving_config=_non_m0_config(),
            diag_state=diag_state,
        )

    target_step = int(args.target_step)
    dense_step_logits = np.asarray(dense_capture["step_logits"][target_step], dtype=np.float32)
    real_mixed_step_logits = np.asarray(real_mixed_capture["step_logits"][target_step], dtype=np.float32)
    non_m0_step_logits = np.asarray(non_m0_capture["step_logits"][target_step], dtype=np.float32)

    per_layer_deltas = _per_layer_deltas(
        non_m0_capture["capture_records"][target_step],
        real_mixed_capture["capture_records"][target_step],
    )
    first_nonzero_layer = _first_nonzero_output_delta(per_layer_deltas)
    real_mixed_final_mix = [
        entry
        for entry in diag_state.final_mix_entries
        if entry["lane"] == "real_mixed" and int(entry["step_index"]) == target_step
    ]
    layer_decode_entries = [
        entry
        for entry in diag_state.layer_decode_entries
        if int(entry["step_index"]) == target_step
    ]

    captured_final_mix_deltas = [
        float(entry["context_ref_max_abs_delta"])
        for entry in real_mixed_final_mix
        if entry.get("context_ref_max_abs_delta") is not None
    ]
    max_captured_final_mix_delta = max(captured_final_mix_deltas, default=0.0)
    final_mix_ref_exact = bool(captured_final_mix_deltas) and float(max_captured_final_mix_delta) <= 1e-5
    if not real_mixed_final_mix or not captured_final_mix_deltas:
        interpretation = (
            "The default same-tree run reproduces the real-mixed `15`/`16` flip and localizes "
            f"the first real-mixed vs non-M0 output drift to full-attention layer `{first_nonzero_layer}`. "
            "This pass did not include per-head final-mix input capture, so it does not by itself "
            "implicate the final_mix helper."
        )
    elif first_nonzero_layer is None and final_mix_ref_exact:
        interpretation = (
            "Real mixed and non-M0 land on the same effective step-"
            f"{target_step} distribution on this run. The recorded final-mix path matches "
            "a float32 softmax+matmul reference on the same inputs, so the residual class is "
            "consistent with backend tie resolution rather than a final_mix bug."
        )
    elif final_mix_ref_exact:
        interpretation = (
            "The first real-mixed vs non-M0 output drift appears at full-attention layer "
            f"`{first_nonzero_layer}`, but the captured direct-M0/final-mix context matches a float32 "
            f"reference on the same inputs to within `{max_captured_final_mix_delta:.8f}`. That points away from the "
            "final_mix kernel and toward tiny upstream mixed-path numeric drift before argmax."
        )
    else:
        interpretation = (
            "The tied-step difference is not explained cleanly by upstream logits alone: "
            "the final-mix helper shows measurable deviation from a float32 reference on "
            "its own inputs, so final_mix remains implicated."
        )

    payload = {
        "force_stream_attn_capture": bool(args.force_stream_attn_capture),
        "prompt_path": str(prompt_path),
        "prompt_length": int(args.prompt_length),
        "decode_steps": int(args.decode_steps),
        "target_step": target_step,
        "lanes": {
            "dense": {
                "generated_ids": dense_generated_ids,
                "target_step_readout": _token_readout(dense_step_logits, token_ids=[15, 16]),
            },
            "real_mixed": {
                "generated_ids": real_mixed_capture["generated_ids"],
                "target_step_readout": _token_readout(real_mixed_step_logits, token_ids=[15, 16]),
            },
            "non_m0": {
                "generated_ids": non_m0_capture["generated_ids"],
                "target_step_readout": _token_readout(non_m0_step_logits, token_ids=[15, 16]),
            },
        },
        "comparisons": {
            "real_mixed_vs_non_m0": {
                "target_step_logit_max_abs_delta": float(
                    np.max(np.abs(real_mixed_step_logits.reshape(-1) - non_m0_step_logits.reshape(-1)))
                ),
                "token15_logit_delta": float(real_mixed_step_logits.reshape(-1)[15] - non_m0_step_logits.reshape(-1)[15]),
                "token16_logit_delta": float(real_mixed_step_logits.reshape(-1)[16] - non_m0_step_logits.reshape(-1)[16]),
                "first_nonzero_output_delta_layer_id": first_nonzero_layer,
                "per_layer_output_deltas": per_layer_deltas,
            },
        },
        "max_captured_final_mix_context_ref_delta": float(max_captured_final_mix_delta),
        "real_mixed_final_mix": real_mixed_final_mix,
        "real_mixed_layer_decode_entries": [
            entry for entry in layer_decode_entries if entry["lane"] == "real_mixed"
        ],
        "non_m0_layer_decode_entries": [
            entry for entry in layer_decode_entries if entry["lane"] == "non_m0"
        ],
        "interpretation": interpretation,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
    def _wrapped_direct_m0(
        *,
        state: Any,
        block_ids: Any,
        query: Any,
        q_head_to_kv_head: np.ndarray,
        query_scale: float,
        config: Any,
        dotcache_config: Any | None = None,
        return_stream_stats: bool = False,
        return_attn_weights: bool = True,
    ):
        result = original_direct_m0(
            state=state,
            block_ids=block_ids,
            query=query,
            q_head_to_kv_head=q_head_to_kv_head,
            query_scale=query_scale,
            config=config,
            dotcache_config=dotcache_config,
            return_stream_stats=return_stream_stats,
            return_attn_weights=return_attn_weights,
        )
        if diag_state.active_step != diag_state.target_step or not bool(return_stream_stats):
            return result
        resolved_block_ids = _resolve_block_ids(block_ids)
        gathered_values = _gather_selected_values_for_block_ids(state, resolved_block_ids)
        q_to_kv = np.asarray(q_head_to_kv_head, dtype=np.int64)
        per_head_logits = result["stream_stats"]["per_head_logits"]
        output = result["output"].detach().to(dtype=torch.float32)
        ref_context = torch.zeros_like(output)
        per_head_entries: list[dict[str, Any]] = []
        for q_head_idx, head_logits in enumerate(per_head_logits):
            if head_logits is None or int(head_logits.numel()) <= 0:
                continue
            logits_f32 = head_logits.detach().to(dtype=torch.float32)
            weights = torch.softmax(logits_f32, dim=-1)
            kv_head_idx = int(q_to_kv[int(q_head_idx)])
            head_context = torch.matmul(weights, gathered_values[kv_head_idx]).to(dtype=torch.float32)
            ref_context[int(q_head_idx)] = head_context
            top_values, top_indices = torch.topk(logits_f32, k=min(4, int(logits_f32.numel())))
            per_head_entries.append(
                {
                    "q_head_id": int(q_head_idx),
                    "kv_head_id": int(kv_head_idx),
                    "top_token_indices": [int(value) for value in top_indices.detach().cpu().tolist()],
                    "top_logits": [float(value) for value in top_values.detach().cpu().tolist()],
                    "weight_sum": float(weights.sum().item()),
                    "top_weight_values": [
                        float(value)
                        for value in torch.topk(weights, k=min(4, int(weights.numel()))).values.detach().cpu().tolist()
                    ],
                }
            )
        context_delta = torch.abs(output - ref_context)
        diag_state.final_mix_entries.append(
            {
                "lane": diag_state.active_lane,
                "step_index": diag_state.active_step,
                "layer_id": diag_state.active_layer,
                "path": "direct_m0_stream_stats",
                "query_scale": float(query_scale),
                "selected_block_ids": [int(value) for value in resolved_block_ids],
                "selected_block_count": int(len(resolved_block_ids)),
                "token_count": int(gathered_values.shape[1]),
                "executed_mode_counts": {
                    key: int(value) for key, value in result.get("executed_mode_counts", {}).items()
                },
                "gathered_values_rms": float(torch.sqrt(torch.mean(torch.square(gathered_values))).item()),
                "gathered_values_max_abs": float(torch.max(torch.abs(gathered_values)).item()),
                "context_ref_max_abs_delta": float(torch.max(context_delta).item()),
                "context_ref_mean_abs_delta": float(torch.mean(context_delta).item()),
                "per_head": per_head_entries,
            }
        )
        return result
