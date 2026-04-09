from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .persistent_types import (
    PersistentFullAttentionLayerState,
    PersistentLayerTelemetry,
    PersistentLinearAttentionLayerState,
    PersistentServingConfig,
    PersistentStepTelemetry,
)


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError("torch is required for the persistent Metal runtime") from exc
    return torch


def _load_torch_functional():
    torch = _load_torch()
    import torch.nn.functional as F

    return torch, F


def _torch_dtype_bytes(dtype: Any) -> int:
    torch = _load_torch()
    probe = torch.empty((), dtype=dtype)
    return int(probe.element_size())


def _clone_tensor_like(value: Any, *, dtype=None, device=None):
    torch = _load_torch()
    if value is None:
        return None
    if torch.is_tensor(value):
        target = value.detach()
        if dtype is not None:
            target = target.to(dtype=dtype)
        if device is not None:
            target = target.to(device=device)
        return target.clone()
    target = torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    return target.clone()


def _nbytes_tensor_like(value: Any) -> int:
    torch = _load_torch()
    if value is None:
        return 0
    if torch.is_tensor(value):
        return int(value.nelement() * value.element_size())
    array = np.asarray(value)
    return int(array.nbytes)


def _sequence_value_or_none(container: Any, index: int) -> Any | None:
    if container is None:
        return None
    try:
        return container[index]
    except Exception:
        return None


def _metal_bindings_available() -> bool:
    try:  # pragma: no cover - depends on optional runtime bindings
        import Foundation  # type: ignore # noqa: F401
        import Metal  # type: ignore # noqa: F401
        import objc  # type: ignore # noqa: F401
    except Exception:
        return False
    return True


class _MetalKernelExecutor:
    def __init__(self) -> None:
        self.source_path = Path(__file__).with_name("persistent_attention.metal")
        self.backend_kind = "metal_pyobjc_stub" if _metal_bindings_available() else "torch_exact_fallback"

    def decode_exact(
        self,
        *,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        q_head_to_kv_head: np.ndarray,
        query_scale: float,
        block_size: int,
    ):
        del block_size
        return _decode_full_attention_exact_torch(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            q_head_to_kv_head=q_head_to_kv_head,
            query_scale=query_scale,
        )


def _decode_full_attention_exact_torch(
    *,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    q_head_to_kv_head: np.ndarray,
    query_scale: float,
):
    torch = _load_torch()
    query_tensor = query.to(dtype=torch.float32)
    key_tensor = key_cache.to(dtype=torch.float32)
    value_tensor = value_cache.to(dtype=torch.float32)
    q_head_to_kv = np.asarray(q_head_to_kv_head, dtype=np.int64)
    output = torch.empty((query_tensor.shape[0], value_tensor.shape[-1]), dtype=torch.float32, device=query_tensor.device)
    for kv_head in sorted(set(int(value) for value in q_head_to_kv.tolist())):
        head_ids = np.flatnonzero(q_head_to_kv == int(kv_head))
        if head_ids.size == 0:
            continue
        q_slice = query_tensor[torch.as_tensor(head_ids, dtype=torch.int64, device=query_tensor.device)]
        k_slice = key_tensor[int(kv_head)]
        v_slice = value_tensor[int(kv_head)]
        logits = torch.matmul(q_slice, k_slice.transpose(0, 1)) * float(query_scale)
        weights = torch.softmax(logits, dim=-1)
        context = torch.matmul(weights.to(dtype=torch.float32), v_slice)
        output[torch.as_tensor(head_ids, dtype=torch.int64, device=query_tensor.device)] = context
    return output


def _build_block_layout(*, token_count: int, block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_token_starts = []
    block_token_counts = []
    metadata_valid = []
    for token_start in range(0, int(token_count), int(block_size)):
        block_token_starts.append(int(token_start))
        block_token_counts.append(int(min(int(block_size), int(token_count) - int(token_start))))
        metadata_valid.append(1.0)
    return (
        np.asarray(block_token_starts, dtype=np.int64),
        np.asarray(block_token_counts, dtype=np.int64),
        np.asarray(metadata_valid, dtype=np.float32),
    )


def _build_block_region_ids(*, num_blocks: int) -> np.ndarray:
    if num_blocks <= 0:
        return np.zeros((0,), dtype=np.int32)
    if num_blocks == 1:
        return np.asarray([1], dtype=np.int32)
    if num_blocks == 2:
        return np.asarray([0, 2], dtype=np.int32)
    region_ids = np.empty((num_blocks,), dtype=np.int32)
    boundaries = np.linspace(0, num_blocks, num=4, dtype=np.int64)
    region_ids[boundaries[0] : boundaries[1]] = 0
    region_ids[boundaries[1] : boundaries[2]] = 1
    region_ids[boundaries[2] : boundaries[3]] = 2
    region_ids[-1] = 2
    return region_ids


def _allocate_full_attention_block_metadata(
    *,
    key_cache: Any,
    value_cache: Any,
    num_blocks: int,
    device: Any,
):
    torch = _load_torch()
    kv_heads = int(key_cache.shape[0])
    head_dim = int(key_cache.shape[-1])
    block_k_center = torch.zeros((num_blocks, kv_heads, head_dim), dtype=torch.float32, device=device)
    block_k_radius = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    block_v_norm_max = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    block_prev_attention_ema = torch.zeros((num_blocks,), dtype=torch.float32, device=device)
    block_k_comp_error = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    return block_k_center, block_k_radius, block_v_norm_max, block_prev_attention_ema, block_k_comp_error


def _recompute_full_attention_block_metadata(
    *,
    state: PersistentFullAttentionLayerState,
    block_indices: list[int] | np.ndarray,
) -> None:
    torch = _load_torch()
    if len(block_indices) == 0:
        return
    for block_idx in [int(i) for i in block_indices]:
        token_start = int(state.block_token_starts[block_idx])
        token_count = int(state.block_token_counts[block_idx])
        if token_count <= 0:
            state.metadata_valid[block_idx] = 0.0
            state.block_k_center[block_idx].zero_()
            state.block_k_radius[block_idx].zero_()
            state.block_v_norm_max[block_idx].zero_()
            state.block_k_comp_error[block_idx].zero_()
            continue
        key_slice = state.key_cache[:, token_start : token_start + token_count, :].to(dtype=torch.float32)
        value_slice = state.value_cache[:, token_start : token_start + token_count, :].to(dtype=torch.float32)
        center = key_slice.mean(dim=1)
        distances = torch.linalg.vector_norm(key_slice - center[:, None, :], dim=-1)
        value_norms = torch.linalg.vector_norm(value_slice, dim=-1)
        state.block_k_center[block_idx].copy_(center)
        state.block_k_radius[block_idx].copy_(distances.max(dim=1).values)
        state.block_v_norm_max[block_idx].copy_(value_norms.max(dim=1).values)
        state.block_k_comp_error[block_idx].zero_()
        state.metadata_valid[block_idx] = 1.0


def _resolve_block_score_inputs(
    *,
    state: PersistentFullAttentionLayerState,
    config: PersistentServingConfig,
    query: Any,
    q_head_to_kv_head: np.ndarray,
    query_scale: float,
):
    torch = _load_torch()
    query_tensor = query.to(dtype=torch.float32)
    query_norm = torch.linalg.vector_norm(query_tensor, dim=-1)
    q_to_kv = np.asarray(q_head_to_kv_head, dtype=np.int64)
    num_blocks = int(len(state.block_token_starts))
    priority_scores = torch.full((num_blocks,), float("-inf"), dtype=torch.float32, device=query_tensor.device)
    upper_bounds = torch.full((num_blocks,), float("-inf"), dtype=torch.float32, device=query_tensor.device)
    block_indices = torch.arange(max(num_blocks, 1), dtype=torch.float32, device=query_tensor.device)
    tail_distance = torch.flip(block_indices, dims=[0])
    recency_decay = max(float(config.full_attention_priority_recency_decay_blocks), 1.0)
    local_recency = torch.exp(-tail_distance / recency_decay)
    prev_weight = float(config.full_attention_priority_prev_attention_weight)
    recency_weight = float(config.full_attention_priority_recency_weight)
    value_weight = float(config.full_attention_priority_value_norm_weight)
    for q_head_idx in range(int(query_tensor.shape[0])):
        kv_head_idx = int(q_to_kv[q_head_idx])
        center = state.block_k_center[:, kv_head_idx, :].to(device=query_tensor.device, dtype=torch.float32)
        radius = state.block_k_radius[:, kv_head_idx].to(device=query_tensor.device, dtype=torch.float32)
        value_norm = state.block_v_norm_max[:, kv_head_idx].to(device=query_tensor.device, dtype=torch.float32)
        comp_error = state.block_k_comp_error[:, kv_head_idx].to(device=query_tensor.device, dtype=torch.float32)
        center_sim = torch.matmul(center, query_tensor[q_head_idx]) * float(query_scale)
        upper = center_sim + query_norm[q_head_idx] * radius * abs(float(query_scale)) + comp_error
        normalized_value_norm = value_norm / value_norm.max().clamp_min(1e-6)
        priority = center_sim
        priority = priority + prev_weight * state.block_prev_attention_ema.to(device=query_tensor.device, dtype=torch.float32)
        priority = priority + recency_weight * local_recency
        priority = priority + value_weight * normalized_value_norm
        priority_scores = torch.maximum(priority_scores, priority)
        upper_bounds = torch.maximum(upper_bounds, upper)
    return priority_scores, upper_bounds


def _resolve_recent_policy(
    *,
    num_blocks: int,
    recent_blocks: int,
    mandatory_recent_blocks: int | None,
) -> tuple[list[int], list[int]]:
    bounded_recent = min(max(int(recent_blocks), 0), int(num_blocks))
    if mandatory_recent_blocks is None:
        bounded_mandatory = bounded_recent
    else:
        bounded_mandatory = min(max(int(mandatory_recent_blocks), 0), bounded_recent)
    recent_ids = [int(num_blocks - bounded_recent + offset) for offset in range(bounded_recent)]
    mandatory_ids = [int(num_blocks - bounded_mandatory + offset) for offset in range(bounded_mandatory)]
    return recent_ids, mandatory_ids


def _mandatory_block_ids(*, num_blocks: int, sink_blocks: int, mandatory_recent_blocks: list[int]) -> list[int]:
    mandatory: set[int] = set()
    for block_id in range(min(max(sink_blocks, 0), num_blocks)):
        mandatory.add(int(block_id))
    for block_id in mandatory_recent_blocks:
        mandatory.add(int(block_id))
    return sorted(mandatory)


def _exploration_block_ids(
    *,
    candidate_block_ids: list[int],
    priority_scores: Any,
    per_region: int,
) -> list[int]:
    if per_region <= 0 or not candidate_block_ids:
        return []
    midpoint = max(1, len(candidate_block_ids) // 2)
    far_region = candidate_block_ids[:midpoint]
    mid_region = candidate_block_ids[midpoint:]
    selected: list[int] = []
    for region_ids in (far_region, mid_region):
        if not region_ids:
            continue
        ranked = sorted(
            region_ids,
            key=lambda block_id: float(priority_scores[int(block_id)].item()),
            reverse=True,
        )
        selected.extend(int(block_id) for block_id in ranked[:per_region])
    return sorted(set(selected))


def _rank_optional_block_ids(
    *,
    candidate_block_ids: list[int],
    priority_scores: Any,
    upper_bounds: Any,
    use_upper_bounds_first: bool,
) -> list[int]:
    if not candidate_block_ids:
        return []
    if bool(use_upper_bounds_first):
        return sorted(
            candidate_block_ids,
            key=lambda block_id: (
                float(upper_bounds[int(block_id)].item()),
                float(priority_scores[int(block_id)].item()),
            ),
            reverse=True,
        )
    return sorted(
        candidate_block_ids,
        key=lambda block_id: float(priority_scores[int(block_id)].item()),
        reverse=True,
    )


def _select_diverse_block_ids(
    *,
    ranked_candidate_ids: list[int],
    primary_scores: Any,
    secondary_scores: Any,
    count: int,
    seed_block_ids: list[int],
    diversity_weight: float,
    diversity_radius: int,
) -> list[int]:
    if count <= 0 or not ranked_candidate_ids:
        return []
    if float(diversity_weight) <= 0.0 or int(diversity_radius) <= 0:
        return [int(block_id) for block_id in ranked_candidate_ids[:count]]
    selected: list[int] = []
    selected_anchor_ids = [int(block_id) for block_id in seed_block_ids]
    remaining = [int(block_id) for block_id in ranked_candidate_ids]
    radius = max(int(diversity_radius), 1)

    def _distance_penalty(block_id: int, anchor_ids: list[int]) -> float:
        if not anchor_ids:
            return 0.0
        min_distance = min(abs(int(block_id) - int(anchor_id)) for anchor_id in anchor_ids)
        if min_distance >= radius:
            return 0.0
        return float(radius - min_distance) / float(radius)

    while remaining and len(selected) < int(count):
        best_block_id = max(
            remaining,
            key=lambda block_id: (
                float(primary_scores[int(block_id)].item())
                - float(diversity_weight) * _distance_penalty(int(block_id), selected_anchor_ids + selected),
                float(primary_scores[int(block_id)].item()),
                float(secondary_scores[int(block_id)].item()),
            ),
        )
        selected.append(int(best_block_id))
        remaining.remove(int(best_block_id))
    return selected


def _select_optional_block_ids(
    *,
    candidate_block_ids: list[int],
    region_ids: Any,
    priority_scores: Any,
    upper_bounds: Any,
    top_k: int,
    use_upper_bounds_first: bool,
    upper_bound_quota: int,
    far_quota: int,
    mid_quota: int,
    near_quota: int,
    seed_block_ids: list[int],
    diversity_weight: float,
    diversity_radius: int,
) -> list[int]:
    if top_k <= 0 or not candidate_block_ids:
        return []
    ranked_by_upper = _rank_optional_block_ids(
        candidate_block_ids=candidate_block_ids,
        priority_scores=priority_scores,
        upper_bounds=upper_bounds,
        use_upper_bounds_first=True,
    )
    ranked_by_priority = _rank_optional_block_ids(
        candidate_block_ids=candidate_block_ids,
        priority_scores=priority_scores,
        upper_bounds=upper_bounds,
        use_upper_bounds_first=False,
    )
    if bool(use_upper_bounds_first):
        return _select_diverse_block_ids(
            ranked_candidate_ids=ranked_by_upper,
            primary_scores=upper_bounds,
            secondary_scores=priority_scores,
            count=int(top_k),
            seed_block_ids=seed_block_ids,
            diversity_weight=float(diversity_weight),
            diversity_radius=int(diversity_radius),
        )

    reserved = max(0, min(int(upper_bound_quota), int(top_k), len(ranked_by_upper)))
    selected: list[int] = _select_diverse_block_ids(
        ranked_candidate_ids=ranked_by_upper,
        primary_scores=upper_bounds,
        secondary_scores=priority_scores,
        count=reserved,
        seed_block_ids=seed_block_ids,
        diversity_weight=float(diversity_weight),
        diversity_radius=int(diversity_radius),
    )
    selected_set = set(selected)
    region_quotas = {
        0: max(0, int(far_quota)),
        1: max(0, int(mid_quota)),
        2: max(0, int(near_quota)),
    }
    for region_id in (0, 1, 2):
        quota = min(region_quotas[region_id], max(int(top_k) - len(selected), 0))
        if quota <= 0:
            continue
        region_ranked = [
            int(block_id)
            for block_id in ranked_by_priority
            if int(block_id) not in selected_set and int(region_ids[int(block_id)]) == int(region_id)
        ]
        region_selected = _select_diverse_block_ids(
            ranked_candidate_ids=region_ranked,
            primary_scores=priority_scores,
            secondary_scores=upper_bounds,
            count=quota,
            seed_block_ids=seed_block_ids + selected,
            diversity_weight=float(diversity_weight),
            diversity_radius=int(diversity_radius),
        )
        for block_id in region_selected:
            selected.append(block_id)
            selected_set.add(block_id)
        if len(selected) >= int(top_k):
            return selected[: int(top_k)]
    spill_ranked = [int(block_id) for block_id in ranked_by_priority if int(block_id) not in selected_set]
    spill_selected = _select_diverse_block_ids(
        ranked_candidate_ids=spill_ranked,
        primary_scores=priority_scores,
        secondary_scores=upper_bounds,
        count=max(int(top_k) - len(selected), 0),
        seed_block_ids=seed_block_ids + selected,
        diversity_weight=float(diversity_weight),
        diversity_radius=int(diversity_radius),
    )
    for block_id in spill_selected:
        selected.append(int(block_id))
        selected_set.add(int(block_id))
    return selected


def _gather_selected_block_tensors(
    *,
    state: PersistentFullAttentionLayerState,
    block_ids: list[int],
):
    torch = _load_torch()
    if not block_ids:
        raise ValueError("selected block ids must not be empty")
    key_slices = []
    value_slices = []
    token_counts: list[int] = []
    for block_id in block_ids:
        token_start = int(state.block_token_starts[block_id])
        token_count = int(state.block_token_counts[block_id])
        token_counts.append(token_count)
        key_slices.append(state.key_cache[:, token_start : token_start + token_count, :])
        value_slices.append(state.value_cache[:, token_start : token_start + token_count, :])
    gathered_keys = torch.cat(key_slices, dim=1)
    gathered_values = torch.cat(value_slices, dim=1)
    return gathered_keys, gathered_values, token_counts


def _update_block_prev_attention_ema(
    *,
    state: PersistentFullAttentionLayerState,
    selected_block_ids: list[int],
    selected_block_token_counts: list[int],
    attn_weights: Any,
    decay: float = 0.9,
) -> None:
    torch = _load_torch()
    if attn_weights is None:
        return
    state.block_prev_attention_ema.mul_(float(decay))
    weights = attn_weights.to(dtype=torch.float32)
    if weights.ndim == 4:
        collapsed = weights.mean(dim=(0, 2))
    elif weights.ndim == 3:
        collapsed = weights.mean(dim=0)
    else:
        collapsed = weights.reshape(-1)
    offset = 0
    for block_id, token_count in zip(selected_block_ids, selected_block_token_counts):
        block_mass = collapsed[offset : offset + int(token_count)].sum()
        state.block_prev_attention_ema[int(block_id)] += (1.0 - float(decay)) * block_mass.to(
            dtype=state.block_prev_attention_ema.dtype,
            device=state.block_prev_attention_ema.device,
        )
        offset += int(token_count)


@dataclass(slots=True)
class PersistentFullAttentionState:
    device: Any
    config: PersistentServingConfig
    q_head_to_kv_head: np.ndarray
    layers: dict[int, PersistentFullAttentionLayerState]
    telemetry: PersistentStepTelemetry
    executor: _MetalKernelExecutor

    @classmethod
    def from_prefill_tensors(
        cls,
        *,
        prefill_tensors: dict[int, tuple[Any, Any]],
        device: Any,
        q_head_to_kv_head: np.ndarray,
        config: PersistentServingConfig,
    ) -> "PersistentFullAttentionState":
        torch = _load_torch()
        resolved_device = device
        telemetry = PersistentStepTelemetry()
        executor = _MetalKernelExecutor()
        telemetry.backend_kind = executor.backend_kind
        layers: dict[int, PersistentFullAttentionLayerState] = {}
        for layer_id, (layer_keys, layer_values) in prefill_tensors.items():
            keys = _clone_tensor_like(layer_keys, dtype=torch.float32, device=resolved_device)
            values = _clone_tensor_like(layer_values, dtype=torch.float32, device=resolved_device)
            if keys.ndim != 4 or values.ndim != 4:
                raise ValueError("persistent full-attention prefill tensors must have shape [batch, kv_heads, seq, head_dim]")
            if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
                raise ValueError("persistent full-attention runtime only supports batch=1")
            kv_keys = keys[0].contiguous()
            kv_values = values[0].contiguous()
            token_count = int(kv_keys.shape[1])
            block_token_starts, block_token_counts, metadata_valid = _build_block_layout(
                token_count=token_count,
                block_size=int(config.block_size),
            )
            num_blocks = int(len(block_token_starts))
            block_k_center, block_k_radius, block_v_norm_max, block_prev_attention_ema, block_k_comp_error = (
                _allocate_full_attention_block_metadata(
                    key_cache=kv_keys,
                    value_cache=kv_values,
                    num_blocks=num_blocks,
                    device=resolved_device,
                )
            )
            layers[int(layer_id)] = PersistentFullAttentionLayerState(
                layer_id=int(layer_id),
                key_cache=kv_keys,
                value_cache=kv_values,
                block_token_starts=block_token_starts,
                block_token_counts=block_token_counts,
                block_k_center=block_k_center,
                block_k_radius=block_k_radius,
                block_v_norm_max=block_v_norm_max,
                block_prev_attention_ema=block_prev_attention_ema,
                block_region_ids=_build_block_region_ids(num_blocks=num_blocks),
                block_k_comp_error=block_k_comp_error,
                metadata_valid=metadata_valid,
            )
            _recompute_full_attention_block_metadata(
                state=layers[int(layer_id)],
                block_indices=np.arange(num_blocks, dtype=np.int64),
            )
        return cls(
            device=resolved_device,
            config=config,
            q_head_to_kv_head=np.asarray(q_head_to_kv_head, dtype=np.int32).copy(),
            layers=layers,
            telemetry=telemetry,
            executor=executor,
        )

    def append_step(self, layer_id: int, key_step: Any, value_step: Any, token_index: int) -> None:
        torch = _load_torch()
        state = self.layers[int(layer_id)]
        key_tensor = _clone_tensor_like(key_step, dtype=torch.float32, device=self.device)
        value_tensor = _clone_tensor_like(value_step, dtype=torch.float32, device=self.device)
        if key_tensor.ndim == 2:
            key_tensor = key_tensor.unsqueeze(1)
        if value_tensor.ndim == 2:
            value_tensor = value_tensor.unsqueeze(1)
        if key_tensor.ndim != 3 or value_tensor.ndim != 3:
            raise ValueError("persistent full-attention append expects [kv_heads, 1, head_dim] or [kv_heads, head_dim]")
        start = time.perf_counter()
        state.key_cache = torch.cat([state.key_cache, key_tensor], dim=1)
        state.value_cache = torch.cat([state.value_cache, value_tensor], dim=1)
        state.append_count += 1
        token_count = int(state.key_cache.shape[1])
        block_token_starts, block_token_counts, metadata_valid = _build_block_layout(
            token_count=token_count,
            block_size=int(self.config.block_size),
        )
        previous_num_blocks = int(len(state.block_token_starts))
        new_num_blocks = int(len(block_token_starts))
        state.block_token_starts = block_token_starts
        state.block_token_counts = block_token_counts
        state.metadata_valid = metadata_valid
        if new_num_blocks != previous_num_blocks:
            (
                state.block_k_center,
                state.block_k_radius,
                state.block_v_norm_max,
                state.block_prev_attention_ema,
                state.block_k_comp_error,
            ) = _allocate_full_attention_block_metadata(
                key_cache=state.key_cache,
                value_cache=state.value_cache,
                num_blocks=new_num_blocks,
                device=self.device,
            )
            state.block_region_ids = _build_block_region_ids(num_blocks=new_num_blocks)
            recompute_block_indices = np.arange(new_num_blocks, dtype=np.int64)
        else:
            state.block_region_ids = _build_block_region_ids(num_blocks=new_num_blocks)
            recompute_block_indices = np.asarray([new_num_blocks - 1], dtype=np.int64)
        _recompute_full_attention_block_metadata(
            state=state,
            block_indices=recompute_block_indices,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.append_update_ms_total += elapsed_ms
        layer_telemetry = self.telemetry.require_layer(int(layer_id))
        layer_telemetry.append_ms_total += elapsed_ms
        layer_telemetry.mutation_count += 1
        del token_index

    def score_blocks(self, layer_id: int, query: Any, *, query_scale: float) -> dict[str, Any]:
        state = self.layers[int(layer_id)]
        priority_scores, upper_bounds = _resolve_block_score_inputs(
            state=state,
            config=self.config,
            query=query,
            q_head_to_kv_head=self.q_head_to_kv_head,
            query_scale=float(query_scale),
        )
        return {
            "priority_scores": priority_scores,
            "upper_bounds": upper_bounds,
        }

    def select_blocks(self, layer_id: int, query: Any, *, query_scale: float) -> dict[str, Any]:
        state = self.layers[int(layer_id)]
        num_blocks = int(len(state.block_token_starts))
        score_result = self.score_blocks(layer_id, query, query_scale=query_scale)
        priority_scores = score_result["priority_scores"]
        upper_bounds = score_result["upper_bounds"]
        recent_ids, mandatory_recent_ids = _resolve_recent_policy(
            num_blocks=num_blocks,
            recent_blocks=int(self.config.full_attention_recent_block_count),
            mandatory_recent_blocks=self.config.full_attention_mandatory_recent_block_count,
        )
        mandatory_ids = _mandatory_block_ids(
            num_blocks=num_blocks,
            sink_blocks=int(self.config.full_attention_sink_block_count),
            mandatory_recent_blocks=mandatory_recent_ids,
        )
        selected_ids: set[int] = set(mandatory_ids)
        soft_recent_ids = [block_id for block_id in recent_ids if block_id not in selected_ids]
        remaining_ids = [block_id for block_id in range(num_blocks) if block_id not in selected_ids]
        exploration_ids = _exploration_block_ids(
            candidate_block_ids=remaining_ids,
            priority_scores=priority_scores,
            per_region=int(self.config.full_attention_exploration_blocks_per_region),
        )
        selected_ids.update(exploration_ids)
        optional_ids: list[int] = []
        if bool(self.config.enable_priority) and int(self.config.full_attention_optional_top_k) > 0:
            soft_recent_set = set(soft_recent_ids)
            optional_candidates = [block_id for block_id in soft_recent_ids if block_id not in selected_ids]
            optional_candidates.extend(
                block_id
                for block_id in remaining_ids
                if block_id not in selected_ids and block_id not in soft_recent_set
            )
            optional_ids = _select_optional_block_ids(
                candidate_block_ids=optional_candidates,
                region_ids=state.block_region_ids,
                priority_scores=priority_scores,
                upper_bounds=upper_bounds,
                top_k=int(self.config.full_attention_optional_top_k),
                use_upper_bounds_first=bool(self.config.full_attention_optional_use_upper_bounds_first),
                upper_bound_quota=int(self.config.full_attention_optional_upper_bound_quota),
                far_quota=int(self.config.full_attention_optional_far_quota),
                mid_quota=int(self.config.full_attention_optional_mid_quota),
                near_quota=int(self.config.full_attention_optional_near_quota),
                seed_block_ids=sorted(selected_ids),
                diversity_weight=float(self.config.full_attention_optional_diversity_weight),
                diversity_radius=int(self.config.full_attention_optional_diversity_radius),
            )
            selected_ids.update(optional_ids)
        else:
            optional_ids = [block_id for block_id in remaining_ids if block_id not in selected_ids]
            selected_ids.update(optional_ids)
        selected_block_ids = sorted(selected_ids)
        return {
            "selected_block_ids": selected_block_ids,
            "mandatory_block_ids": mandatory_ids,
            "soft_recent_block_ids": soft_recent_ids,
            "exploration_block_ids": exploration_ids,
            "optional_block_ids": optional_ids,
            "priority_scores": priority_scores,
            "upper_bounds": upper_bounds,
        }

    def gather_selected_blocks(self, layer_id: int, block_ids: list[int]):
        state = self.layers[int(layer_id)]
        return _gather_selected_block_tensors(state=state, block_ids=block_ids)

    def update_block_attention_ema(
        self,
        layer_id: int,
        *,
        selected_block_ids: list[int],
        selected_block_token_counts: list[int],
        attn_weights: Any,
    ) -> None:
        state = self.layers[int(layer_id)]
        _update_block_prev_attention_ema(
            state=state,
            selected_block_ids=selected_block_ids,
            selected_block_token_counts=selected_block_token_counts,
            attn_weights=attn_weights,
        )

    def decode_layer(self, layer_id: int, query: Any, *, query_scale: float):
        state = self.layers[int(layer_id)]
        start = time.perf_counter()
        context = self.executor.decode_exact(
            query=query,
            key_cache=state.key_cache,
            value_cache=state.value_cache,
            q_head_to_kv_head=self.q_head_to_kv_head,
            query_scale=float(query_scale),
            block_size=int(self.config.block_size),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.full_attention_step_ms_total += elapsed_ms
        layer_telemetry = self.telemetry.require_layer(int(layer_id))
        layer_telemetry.decode_ms_total += elapsed_ms
        return context

    def resident_nbytes(self) -> int:
        total = 0
        for state in self.layers.values():
            total += _nbytes_tensor_like(state.key_cache)
            total += _nbytes_tensor_like(state.value_cache)
        return int(total)

    def summary(self) -> dict[str, Any]:
        return {
            "persistent_full_attention_resident_bytes": int(self.resident_nbytes()),
            "persistent_runtime_backend_kind": str(self.telemetry.backend_kind),
            "persistent_host_to_device_bytes_after_prefill": int(self.telemetry.host_to_device_bytes_after_prefill),
            "persistent_full_attention_step_ms_total": float(self.telemetry.full_attention_step_ms_total),
            "persistent_append_update_ms_total": float(self.telemetry.append_update_ms_total),
            "persistent_runtime_mandatory_recent_block_count": (
                None
                if self.config.full_attention_mandatory_recent_block_count is None
                else int(self.config.full_attention_mandatory_recent_block_count)
            ),
            "persistent_full_attention_append_counts_by_layer": {
                str(layer_id): int(state.append_count) for layer_id, state in sorted(self.layers.items())
            },
            "persistent_full_attention_block_count_by_layer": {
                str(layer_id): int(len(state.block_token_starts)) for layer_id, state in sorted(self.layers.items())
            },
            "persistent_full_attention_metadata_valid_blocks_by_layer": {
                str(layer_id): int(np.count_nonzero(state.metadata_valid)) for layer_id, state in sorted(self.layers.items())
            },
            "persistent_full_attention_decode_ms_total_by_layer": {
                str(layer_id): float(self.telemetry.require_layer(layer_id).decode_ms_total)
                for layer_id in sorted(self.layers)
            },
            "persistent_append_update_ms_total_by_layer": {
                str(layer_id): float(self.telemetry.require_layer(layer_id).append_ms_total)
                for layer_id in sorted(self.layers)
            },
        }


@dataclass(slots=True)
class PersistentLinearAttentionState:
    device: Any
    layers: dict[int, PersistentLinearAttentionLayerState]
    telemetry: PersistentStepTelemetry

    @classmethod
    def from_native_cache(
        cls,
        *,
        cache: Any,
        layer_ids: list[int],
        device: Any,
        telemetry: PersistentStepTelemetry,
    ) -> "PersistentLinearAttentionState":
        layers: dict[int, PersistentLinearAttentionLayerState] = {}
        conv_states = getattr(cache, "conv_states", None)
        recurrent_states = getattr(cache, "recurrent_states", None)
        for layer_id in layer_ids:
            layer_cache = _sequence_value_or_none(getattr(cache, "layers", None), int(layer_id))
            conv_state_value = _sequence_value_or_none(conv_states, int(layer_id))
            if conv_state_value is None and layer_cache is not None:
                conv_state_value = getattr(layer_cache, "conv_states", None)
            recurrent_state_value = _sequence_value_or_none(recurrent_states, int(layer_id))
            if recurrent_state_value is None and layer_cache is not None:
                recurrent_state_value = getattr(layer_cache, "recurrent_states", None)
            conv_state = _clone_tensor_like(conv_state_value, device=device)
            recurrent_state = _clone_tensor_like(recurrent_state_value, device=device)
            layers[int(layer_id)] = PersistentLinearAttentionLayerState(
                layer_id=int(layer_id),
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                has_previous_state=bool(getattr(layer_cache, "has_previous_state", False)),
            )
        return cls(device=device, layers=layers, telemetry=telemetry)

    def decode_layer(
        self,
        layer_id: int,
        *,
        layer_module: Any,
        hidden_states: Any,
        attention_mask: Any | None,
    ):
        torch, F = _load_torch_functional()
        state = self.layers[int(layer_id)]
        start = time.perf_counter()

        if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] == hidden_states.shape[1]:
            masked_hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        else:
            masked_hidden_states = hidden_states

        batch_size, seq_len, _ = masked_hidden_states.shape
        mixed_qkv = layer_module.in_proj_qkv(masked_hidden_states).transpose(1, 2)
        z = layer_module.in_proj_z(masked_hidden_states).reshape(batch_size, seq_len, -1, layer_module.head_v_dim)
        b = layer_module.in_proj_b(masked_hidden_states)
        a = layer_module.in_proj_a(masked_hidden_states)

        use_precomputed_states = (
            bool(state.has_previous_state)
            and seq_len == 1
            and state.conv_state is not None
            and state.recurrent_state is not None
        )

        if use_precomputed_states:
            conv_state = state.conv_state.to(device=masked_hidden_states.device)
            recurrent_state = state.recurrent_state.to(device=masked_hidden_states.device)
            mixed_qkv = layer_module.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                layer_module.conv1d.weight.squeeze(1),
                layer_module.conv1d.bias,
                layer_module.activation,
            )
            state.conv_state = _clone_tensor_like(conv_state, device=self.device)
        else:
            conv_state = F.pad(mixed_qkv, (layer_module.conv_kernel_size - mixed_qkv.shape[-1], 0))
            state.conv_state = _clone_tensor_like(conv_state, device=self.device)
            if layer_module.causal_conv1d_fn is not None:
                mixed_qkv = layer_module.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=layer_module.conv1d.weight.squeeze(1),
                    bias=layer_module.conv1d.bias,
                    activation=layer_module.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(layer_module.conv1d(mixed_qkv)[:, :, :seq_len])
            recurrent_state = state.recurrent_state.to(device=masked_hidden_states.device) if state.recurrent_state is not None else None

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [layer_module.key_dim, layer_module.key_dim, layer_module.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, layer_module.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, layer_module.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, layer_module.head_v_dim)

        beta = b.sigmoid()
        g = -layer_module.A_log.float().exp() * F.softplus(a.float() + layer_module.dt_bias)
        if layer_module.num_v_heads // layer_module.num_k_heads > 1:
            query = query.repeat_interleave(layer_module.num_v_heads // layer_module.num_k_heads, dim=2)
            key = key.repeat_interleave(layer_module.num_v_heads // layer_module.num_k_heads, dim=2)

        if use_precomputed_states:
            core_attn_out, last_recurrent_state = layer_module.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = layer_module.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        if last_recurrent_state is not None:
            state.recurrent_state = _clone_tensor_like(last_recurrent_state, device=self.device)
        state.has_previous_state = True
        state.direct_compute_count += 1

        core_attn_out = core_attn_out.reshape(-1, layer_module.head_v_dim)
        z = z.reshape(-1, layer_module.head_v_dim)
        core_attn_out = layer_module.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = layer_module.out_proj(core_attn_out)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.linear_attention_step_ms_total += elapsed_ms
        layer_telemetry = self.telemetry.require_layer(int(layer_id))
        layer_telemetry.linear_ms_total += elapsed_ms
        layer_telemetry.mutation_count += 1
        return output

    def sync_layer_into_cache(self, cache_params: Any, layer_id: int) -> None:
        state = self.layers[int(layer_id)]
        start = time.perf_counter()
        if hasattr(cache_params, "conv_states") and int(layer_id) < len(cache_params.conv_states) and state.conv_state is not None:
            cache_params.conv_states[int(layer_id)] = _clone_tensor_like(state.conv_state, device=state.conv_state.device)
        if hasattr(cache_params, "recurrent_states") and int(layer_id) < len(cache_params.recurrent_states) and state.recurrent_state is not None:
            cache_params.recurrent_states[int(layer_id)] = _clone_tensor_like(state.recurrent_state, device=state.recurrent_state.device)
        state.sync_into_cache_count += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        layer_telemetry = self.telemetry.require_layer(int(layer_id))
        layer_telemetry.update_ms_total += elapsed_ms

    def sync_layer_from_cache(self, cache_params: Any, layer_id: int) -> None:
        state = self.layers[int(layer_id)]
        start = time.perf_counter()
        if hasattr(cache_params, "conv_states") and int(layer_id) < len(cache_params.conv_states):
            state.conv_state = _clone_tensor_like(cache_params.conv_states[int(layer_id)], device=self.device)
        if hasattr(cache_params, "recurrent_states") and int(layer_id) < len(cache_params.recurrent_states):
            state.recurrent_state = _clone_tensor_like(cache_params.recurrent_states[int(layer_id)], device=self.device)
        state.sync_from_cache_count += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.append_update_ms_total += elapsed_ms
        layer_telemetry = self.telemetry.require_layer(int(layer_id))
        layer_telemetry.update_ms_total += elapsed_ms
        layer_telemetry.mutation_count += 1

    def resident_nbytes(self) -> int:
        total = 0
        for state in self.layers.values():
            total += _nbytes_tensor_like(state.conv_state)
            total += _nbytes_tensor_like(state.recurrent_state)
        return int(total)

    def summary(self) -> dict[str, Any]:
        return {
            "persistent_linear_resident_bytes": int(self.resident_nbytes()),
            "persistent_linear_attention_step_ms_total": float(self.telemetry.linear_attention_step_ms_total),
            "persistent_linear_state_sync_into_cache_count_by_layer": {
                str(layer_id): int(state.sync_into_cache_count) for layer_id, state in sorted(self.layers.items())
            },
            "persistent_linear_state_sync_from_cache_count_by_layer": {
                str(layer_id): int(state.sync_from_cache_count) for layer_id, state in sorted(self.layers.items())
            },
            "persistent_linear_direct_compute_count_by_layer": {
                str(layer_id): int(state.direct_compute_count) for layer_id, state in sorted(self.layers.items())
            },
            "persistent_linear_mutation_counts_by_layer": {
                str(layer_id): int(self.telemetry.require_layer(layer_id).mutation_count)
                for layer_id in sorted(self.layers)
            },
        }


@dataclass(slots=True)
class PersistentHybridRuntimeState:
    native_state: Any
    full_attention: PersistentFullAttentionState
    linear_attention: PersistentLinearAttentionState
    config: PersistentServingConfig

    @classmethod
    def from_post_handoff_cache(
        cls,
        *,
        native_state: Any,
        prefill_tensors: dict[int, tuple[Any, Any]],
        linear_layer_ids: list[int],
        q_head_to_kv_head: np.ndarray,
        device: Any,
        config: PersistentServingConfig,
    ) -> "PersistentHybridRuntimeState":
        telemetry = PersistentStepTelemetry()
        full_attention = PersistentFullAttentionState.from_prefill_tensors(
            prefill_tensors=prefill_tensors,
            device=device,
            q_head_to_kv_head=q_head_to_kv_head,
            config=config,
        )
        telemetry.backend_kind = full_attention.telemetry.backend_kind
        telemetry.host_to_device_bytes_after_prefill = full_attention.telemetry.host_to_device_bytes_after_prefill
        linear_attention = PersistentLinearAttentionState.from_native_cache(
            cache=native_state.past_key_values,
            layer_ids=linear_layer_ids,
            device=device,
            telemetry=telemetry,
        )
        full_attention.telemetry = telemetry
        return cls(
            native_state=native_state,
            full_attention=full_attention,
            linear_attention=linear_attention,
            config=config,
        )

    @property
    def model_past_key_values(self) -> Any:
        return self.native_state.past_key_values

    def advance(self, past_key_values: Any) -> None:
        self.native_state.refresh(past_key_values)

    def decode_full_attention_layer(self, layer_id: int, query: Any, *, query_scale: float):
        return self.full_attention.decode_layer(layer_id, query, query_scale=query_scale)

    def append_full_attention_step(self, layer_id: int, key_step: Any, value_step: Any, token_index: int) -> None:
        self.full_attention.append_step(layer_id, key_step, value_step, token_index)

    def score_full_attention_blocks(self, layer_id: int, query: Any, *, query_scale: float) -> dict[str, Any]:
        return self.full_attention.score_blocks(layer_id, query, query_scale=query_scale)

    def select_full_attention_blocks(self, layer_id: int, query: Any, *, query_scale: float) -> dict[str, Any]:
        return self.full_attention.select_blocks(layer_id, query, query_scale=query_scale)

    def gather_full_attention_selected_blocks(self, layer_id: int, block_ids: list[int]):
        return self.full_attention.gather_selected_blocks(layer_id, block_ids)

    def update_full_attention_block_attention_ema(
        self,
        layer_id: int,
        *,
        selected_block_ids: list[int],
        selected_block_token_counts: list[int],
        attn_weights: Any,
    ) -> None:
        self.full_attention.update_block_attention_ema(
            layer_id,
            selected_block_ids=selected_block_ids,
            selected_block_token_counts=selected_block_token_counts,
            attn_weights=attn_weights,
        )

    def sync_linear_layer_into_cache(self, cache_params: Any, layer_id: int) -> None:
        self.linear_attention.sync_layer_into_cache(cache_params, layer_id)

    def sync_linear_layer_from_cache(self, cache_params: Any, layer_id: int) -> None:
        self.linear_attention.sync_layer_from_cache(cache_params, layer_id)

    def decode_linear_attention_layer(
        self,
        layer_id: int,
        *,
        layer_module: Any,
        hidden_states: Any,
        attention_mask: Any | None,
    ):
        return self.linear_attention.decode_layer(
            layer_id,
            layer_module=layer_module,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    def summary(self) -> dict[str, Any]:
        result = self.native_state.summary()
        result.update(
            {
                "persistent_hybrid_runtime_ready": True,
                "hybrid_runtime_state_kind": "qwen35_attention_subset_persistent",
                "hybrid_runtime_fixed_resident_layer_ids": self.native_state.fixed_resident_layer_ids,
                "hybrid_runtime_token_growing_layer_ids": self.native_state.token_growing_layer_ids,
                "persistent_runtime_dense_only": bool(self.config.dense_only),
                "persistent_runtime_enable_full_attention_persistent_compute": bool(
                    self.config.enable_full_attention_persistent_compute
                ),
                "persistent_runtime_enable_linear_attention_persistent_compute": bool(
                    self.config.enable_linear_attention_persistent_compute
                ),
                "persistent_runtime_enable_priority": bool(self.config.enable_priority),
                "persistent_runtime_enable_early_exit": bool(self.config.enable_early_exit),
                "persistent_runtime_enable_compression": bool(self.config.enable_compression),
            }
        )
        result.update(self.full_attention.summary())
        result.update(self.linear_attention.summary())
        return result
