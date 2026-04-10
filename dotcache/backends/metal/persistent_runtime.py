from __future__ import annotations

from dataclasses import dataclass, replace
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


def _certify_selected_block_frontier(
    *,
    state: PersistentFullAttentionLayerState,
    query: Any,
    q_head_to_kv_head: np.ndarray,
    query_scale: float,
    selected_block_ids: list[int],
    upper_bounds: Any,
    config: PersistentServingConfig,
) -> dict[str, Any]:
    torch = _load_torch()
    query_tensor = query.to(dtype=torch.float32)
    query_norm = torch.linalg.vector_norm(query_tensor, dim=-1)
    q_to_kv = np.asarray(q_head_to_kv_head, dtype=np.int64)
    selected_ids = [int(block_id) for block_id in selected_block_ids]
    selected_set = set(selected_ids)
    omitted_block_ids = [
        int(block_id) for block_id in range(int(len(state.block_token_starts))) if int(block_id) not in selected_set
    ]
    selected_keys, _selected_values, selected_block_token_counts = _gather_selected_block_tensors(
        state=state,
        block_ids=selected_ids,
    )
    bound_eps = float(config.full_attention_bound_eps)
    mass_eps = float(config.full_attention_mass_eps)
    value_eps = float(config.full_attention_value_eps)
    min_processed_blocks = max(int(config.full_attention_min_processed_blocks), 1)
    metadata_invalid_block_ids = [
        int(block_id)
        for block_id in range(int(len(state.block_token_starts)))
        if float(state.metadata_valid[int(block_id)]) <= 0.0
    ]
    max_bound_excess = 0.0
    per_head: list[dict[str, float | int]] = []
    for q_head_idx in range(int(query_tensor.shape[0])):
        kv_head_idx = int(q_to_kv[q_head_idx])
        q_vec = query_tensor[q_head_idx]
        selected_logits = torch.matmul(selected_keys[kv_head_idx], q_vec) * float(query_scale)
        if int(selected_logits.numel()) > 0:
            m_value = float(selected_logits.max().item())
            shifted = torch.exp(selected_logits - m_value)
            l_value = float(shifted.sum().item())
        else:
            m_value = float("-inf")
            l_value = 0.0
            shifted = None
        offset = 0
        for block_id, token_count in zip(selected_ids, selected_block_token_counts):
            count = int(token_count)
            if count <= 0:
                continue
            block_logits = selected_logits[offset : offset + count]
            if int(block_logits.numel()) > 0:
                block_max = float(block_logits.max().item())
                block_excess = block_max - float(upper_bounds[int(block_id)].item())
                max_bound_excess = max(max_bound_excess, block_excess)
            offset += count
        residual_mass_upper, residual_value_upper = _residual_value_upper_for_blocks(
            state=state,
            block_ids=omitted_block_ids,
            kv_head_idx=kv_head_idx,
            q_vec=q_vec,
            q_norm=float(query_norm[q_head_idx].item()),
            query_scale=float(query_scale),
            m_value=m_value,
            upper_bounds=upper_bounds,
            use_region_caps=bool(config.full_attention_region_residual_caps),
            residual_cluster_count=int(config.full_attention_residual_cluster_count),
        )
        denom = float(l_value + residual_mass_upper)
        beta_upper = float(residual_mass_upper / denom) if denom > 0.0 else 0.0
        delta_upper = float(residual_value_upper / denom) if denom > 0.0 else 0.0
        per_head.append(
            {
                "q_head_id": int(q_head_idx),
                "kv_head_id": int(kv_head_idx),
                "m": float(m_value),
                "l": float(l_value),
                "residual_mass_upper": float(residual_mass_upper),
                "residual_value_upper": float(residual_value_upper),
                "beta_upper": float(beta_upper),
                "delta_upper": float(delta_upper),
            }
        )
    beta_upper = max((float(item["beta_upper"]) for item in per_head), default=0.0)
    delta_upper = max((float(item["delta_upper"]) for item in per_head), default=0.0)
    residual_mass_upper = max((float(item["residual_mass_upper"]) for item in per_head), default=0.0)
    residual_value_upper = max((float(item["residual_value_upper"]) for item in per_head), default=0.0)
    instability_reasons: list[str] = []
    if metadata_invalid_block_ids:
        instability_reasons.append("invalid_metadata")
    if float(max_bound_excess) > float(bound_eps):
        instability_reasons.append("bound_exceeded")
    instability_flag = bool(instability_reasons)
    certified_can_stop = (
        int(len(selected_ids)) >= int(min_processed_blocks)
        and not instability_flag
        and float(beta_upper) < float(mass_eps)
        and float(delta_upper) < float(value_eps)
    )
    fallback_recommended = bool(
        instability_flag
        or (
            len(omitted_block_ids) > 0
            and (
                float(beta_upper) >= float(mass_eps)
                or float(delta_upper) >= float(value_eps)
            )
        )
    )
    certificate = {
        "processed_block_count": int(len(selected_ids)),
        "processed_token_count": int(sum(int(count) for count in selected_block_token_counts)),
        "remaining_block_count": int(len(omitted_block_ids)),
        "remaining_token_count": int(
            sum(int(state.block_token_counts[int(block_id)]) for block_id in omitted_block_ids)
        ),
        "beta_upper": float(beta_upper),
        "delta_upper": float(delta_upper),
        "residual_mass_upper": float(residual_mass_upper),
        "residual_value_upper": float(residual_value_upper),
        "max_bound_excess": float(max(0.0, max_bound_excess)),
        "bound_eps": float(bound_eps),
        "mass_eps": float(mass_eps),
        "value_eps": float(value_eps),
        "min_processed_blocks": int(min_processed_blocks),
        "metadata_invalid_block_ids": [int(block_id) for block_id in metadata_invalid_block_ids],
        "instability_flag": bool(instability_flag),
        "instability_reasons": instability_reasons,
        "certified_can_stop": bool(certified_can_stop),
        "fallback_recommended": bool(fallback_recommended),
        "per_head": per_head,
    }
    state.last_residual_certificate = certificate
    return certificate


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
    key_centroid_count: int,
):
    torch = _load_torch()
    kv_heads = int(key_cache.shape[0])
    head_dim = int(key_cache.shape[-1])
    block_k_center = torch.zeros((num_blocks, kv_heads, head_dim), dtype=torch.float32, device=device)
    block_k_radius = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    block_k_subcenters = torch.zeros(
        (num_blocks, kv_heads, max(int(key_centroid_count), 1), head_dim),
        dtype=torch.float32,
        device=device,
    )
    block_k_subradii = torch.zeros(
        (num_blocks, kv_heads, max(int(key_centroid_count), 1)),
        dtype=torch.float32,
        device=device,
    )
    block_v_center = torch.zeros((num_blocks, kv_heads, head_dim), dtype=torch.float32, device=device)
    block_v_radius = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    block_v_norm_max = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    block_prev_attention_ema = torch.zeros((num_blocks,), dtype=torch.float32, device=device)
    block_k_comp_error = torch.zeros((num_blocks, kv_heads), dtype=torch.float32, device=device)
    return (
        block_k_center,
        block_k_radius,
        block_k_subcenters,
        block_k_subradii,
        block_v_center,
        block_v_radius,
        block_v_norm_max,
        block_prev_attention_ema,
        block_k_comp_error,
    )


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
            state.block_k_subcenters[block_idx].zero_()
            state.block_k_subradii[block_idx].zero_()
            state.block_v_center[block_idx].zero_()
            state.block_v_radius[block_idx].zero_()
            state.block_v_norm_max[block_idx].zero_()
            state.block_k_comp_error[block_idx].zero_()
            continue
        key_slice = state.key_cache[:, token_start : token_start + token_count, :].to(dtype=torch.float32)
        value_slice = state.value_cache[:, token_start : token_start + token_count, :].to(dtype=torch.float32)
        center = key_slice.mean(dim=1)
        distances = torch.linalg.vector_norm(key_slice - center[:, None, :], dim=-1)
        state.block_k_subcenters[block_idx].copy_(center[:, None, :].expand_as(state.block_k_subcenters[block_idx]))
        state.block_k_subradii[block_idx].copy_(distances.max(dim=1).values[:, None].expand_as(state.block_k_subradii[block_idx]))
        centroid_count = int(state.block_k_subcenters.shape[2])
        if centroid_count > 1:
            token_partitions = np.array_split(np.arange(token_count, dtype=np.int64), centroid_count)
            for centroid_idx, token_ids in enumerate(token_partitions):
                if len(token_ids) <= 0:
                    continue
                token_index_tensor = torch.as_tensor(token_ids, dtype=torch.int64, device=key_slice.device)
                sub_key_slice = key_slice.index_select(1, token_index_tensor)
                sub_center = sub_key_slice.mean(dim=1)
                sub_distances = torch.linalg.vector_norm(sub_key_slice - sub_center[:, None, :], dim=-1)
                state.block_k_subcenters[block_idx, :, centroid_idx, :].copy_(sub_center)
                state.block_k_subradii[block_idx, :, centroid_idx].copy_(sub_distances.max(dim=1).values)
        value_center = value_slice.mean(dim=1)
        value_distances = torch.linalg.vector_norm(value_slice - value_center[:, None, :], dim=-1)
        value_norms = torch.linalg.vector_norm(value_slice, dim=-1)
        state.block_k_center[block_idx].copy_(center)
        state.block_k_radius[block_idx].copy_(distances.max(dim=1).values)
        state.block_v_center[block_idx].copy_(value_center)
        state.block_v_radius[block_idx].copy_(value_distances.max(dim=1).values)
        state.block_v_norm_max[block_idx].copy_(value_norms.max(dim=1).values)
        state.block_k_comp_error[block_idx].zero_()
        state.metadata_valid[block_idx] = 1.0


def _evenly_spaced_probe_offsets(*, token_count: int, sample_count: int) -> list[int]:
    if token_count <= 0 or sample_count <= 0:
        return []
    if sample_count >= token_count:
        return list(range(int(token_count)))
    raw_offsets = np.linspace(0, int(token_count) - 1, num=int(sample_count))
    offsets = sorted(set(int(round(float(offset))) for offset in raw_offsets))
    if offsets[0] != 0:
        offsets.insert(0, 0)
    if offsets[-1] != int(token_count) - 1:
        offsets.append(int(token_count) - 1)
    return sorted(set(offsets))


def _sample_probe_refined_upper_bound(
    *,
    state: PersistentFullAttentionLayerState,
    block_id: int,
    query_tensor: Any,
    query_norm: Any,
    q_to_kv: np.ndarray,
    query_scale: float,
    sample_count: int,
) -> float | None:
    torch = _load_torch()
    token_start = int(state.block_token_starts[int(block_id)])
    token_count = int(state.block_token_counts[int(block_id)])
    if token_count <= 0 or sample_count <= 0:
        return None
    probe_offsets = _evenly_spaced_probe_offsets(token_count=token_count, sample_count=sample_count)
    if not probe_offsets:
        return None
    probe_bound = float("-inf")
    unique_kv_heads = sorted(set(int(value) for value in np.asarray(q_to_kv, dtype=np.int64).tolist()))
    for kv_head_idx in unique_kv_heads:
        key_slice = state.key_cache[kv_head_idx, token_start : token_start + token_count, :].to(
            device=query_tensor.device,
            dtype=torch.float32,
        )
        if int(key_slice.shape[0]) == 0:
            continue
        sample_keys = key_slice[
            torch.as_tensor(probe_offsets, dtype=torch.int64, device=query_tensor.device)
        ]
        if int(sample_keys.shape[0]) == 0:
            continue
        nearest_sample_radius = torch.cdist(key_slice, sample_keys).min(dim=1).values.max()
        kv_comp_error = float(state.block_k_comp_error[int(block_id), kv_head_idx].item())
        kv_q_head_ids = np.flatnonzero(q_to_kv == int(kv_head_idx))
        for q_head_idx in kv_q_head_ids.tolist():
            sample_logits = torch.matmul(sample_keys, query_tensor[int(q_head_idx)]) * float(query_scale)
            candidate_upper = float(sample_logits.max().item())
            candidate_upper += float(query_norm[int(q_head_idx)].item()) * float(nearest_sample_radius.item()) * abs(
                float(query_scale)
            )
            candidate_upper += kv_comp_error
            probe_bound = max(probe_bound, candidate_upper)
    if not math.isfinite(probe_bound):
        return None
    return float(probe_bound)


def _cluster_block_ids_by_key_envelope(
    *,
    state: PersistentFullAttentionLayerState,
    block_ids: list[int] | set[int],
    kv_head_idx: int,
    cluster_count: int,
) -> list[list[int]]:
    torch = _load_torch()
    resolved_ids = [int(block_id) for block_id in block_ids if int(state.block_token_counts[int(block_id)]) > 0]
    if not resolved_ids:
        return []
    if cluster_count <= 0 or len(resolved_ids) <= int(cluster_count):
        return [[int(block_id)] for block_id in resolved_ids]
    centers = state.block_k_center[
        torch.as_tensor(resolved_ids, dtype=torch.int64, device=state.block_k_center.device),
        int(kv_head_idx),
        :,
    ].to(dtype=torch.float32)
    distances = torch.cdist(centers, centers)
    seed_positions = [0]
    while len(seed_positions) < min(int(cluster_count), len(resolved_ids)):
        min_distance = distances[:, seed_positions].min(dim=1).values
        next_seed = int(min_distance.argmax().item())
        if next_seed in seed_positions:
            break
        seed_positions.append(next_seed)
    seed_centers = centers[torch.as_tensor(seed_positions, dtype=torch.int64, device=centers.device)]
    assignments = torch.cdist(centers, seed_centers).argmin(dim=1)
    clusters: list[list[int]] = []
    for cluster_idx in range(int(seed_centers.shape[0])):
        member_positions = torch.nonzero(assignments == int(cluster_idx), as_tuple=False).flatten().tolist()
        if not member_positions:
            continue
        clusters.append([int(resolved_ids[position]) for position in member_positions])
    return clusters


def _residual_value_upper_for_blocks(
    *,
    state: PersistentFullAttentionLayerState,
    block_ids: list[int] | set[int],
    kv_head_idx: int,
    q_vec: Any,
    q_norm: float,
    query_scale: float,
    m_value: float,
    upper_bounds: Any,
    use_region_caps: bool,
    residual_cluster_count: int,
) -> tuple[float, float]:
    torch = _load_torch()
    grouped_block_ids: list[list[int]]
    if int(residual_cluster_count) > 0:
        grouped_block_ids = _cluster_block_ids_by_key_envelope(
            state=state,
            block_ids=block_ids,
            kv_head_idx=int(kv_head_idx),
            cluster_count=int(residual_cluster_count),
        )
    else:
        region_groups: dict[int, list[int]] = {}
        for remaining_block_id in block_ids:
            region_groups.setdefault(int(state.block_region_ids[int(remaining_block_id)]), []).append(int(remaining_block_id))
        grouped_block_ids = list(region_groups.values())

    residual_mass_upper = 0.0
    residual_value_upper = 0.0
    q_vec_tensor = q_vec.to(device=state.key_cache.device, dtype=torch.float32)
    use_group_caps = bool(use_region_caps) or int(residual_cluster_count) > 0
    for region_block_ids in grouped_block_ids:
        weighted_center_sum = torch.zeros(
            (int(state.value_cache.shape[-1]),),
            dtype=torch.float32,
            device=state.value_cache.device,
        )
        block_mass_upper = 0.0
        block_radius_upper = 0.0
        block_norm_upper = 0.0
        token_spans: list[tuple[int, int]] = []
        for remaining_block_id in region_block_ids:
            block_token_count = int(state.block_token_counts[int(remaining_block_id)])
            if block_token_count <= 0:
                continue
            token_start = int(state.block_token_starts[int(remaining_block_id)])
            token_spans.append((token_start, block_token_count))
            upper_value = float(upper_bounds[int(remaining_block_id)].item())
            scaled = math.exp(min(upper_value - m_value, 80.0)) if math.isfinite(m_value) else math.exp(
                min(upper_value, 80.0)
            )
            current_block_mass = float(block_token_count) * scaled
            block_mass_upper += current_block_mass
            weighted_center_sum = weighted_center_sum + (
                state.block_v_center[int(remaining_block_id), int(kv_head_idx)].to(dtype=torch.float32)
                * float(current_block_mass)
            )
            block_radius_upper += (
                float(current_block_mass)
                * float(state.block_v_radius[int(remaining_block_id), int(kv_head_idx)].item())
            )
            block_norm_upper += (
                float(current_block_mass)
                * float(state.block_v_norm_max[int(remaining_block_id), int(kv_head_idx)].item())
            )
        block_value_upper = min(
            float(torch.linalg.vector_norm(weighted_center_sum).item()) + float(block_radius_upper),
            float(block_norm_upper),
        )
        if not use_group_caps or not token_spans:
            residual_mass_upper += float(block_mass_upper)
            residual_value_upper += float(block_value_upper)
            continue
        member_ids = [int(block_id) for block_id in region_block_ids]
        member_index_tensor = torch.as_tensor(member_ids, dtype=torch.int64, device=state.block_k_center.device)
        cluster_centers = state.block_k_center[member_index_tensor, int(kv_head_idx), :].to(dtype=torch.float32)
        cluster_radii = state.block_k_radius[member_index_tensor, int(kv_head_idx)].to(dtype=torch.float32)
        cluster_center = cluster_centers.mean(dim=0)
        center_offsets = torch.linalg.vector_norm(cluster_centers - cluster_center[None, :], dim=-1)
        cluster_radius = (center_offsets + cluster_radii).max()
        cluster_upper = float(torch.dot(cluster_center, q_vec_tensor).item()) * float(query_scale)
        cluster_upper += float(q_norm) * float(cluster_radius.item()) * abs(float(query_scale))
        cluster_token_count = int(sum(int(state.block_token_counts[int(block_id)]) for block_id in member_ids))
        cluster_mass_cap = float(cluster_token_count) * (
            math.exp(min(cluster_upper - m_value, 80.0)) if math.isfinite(m_value) else math.exp(min(cluster_upper, 80.0))
        )
        cluster_value_norm_cap = cluster_mass_cap * float(
            state.block_v_norm_max[member_index_tensor, int(kv_head_idx)].max().item()
        )
        residual_mass_upper += min(float(block_mass_upper), float(cluster_mass_cap))
        residual_value_upper += min(float(block_value_upper), float(cluster_value_norm_cap))
    return float(residual_mass_upper), float(residual_value_upper)


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
        subcenters = state.block_k_subcenters[:, kv_head_idx, :, :].to(device=query_tensor.device, dtype=torch.float32)
        subradii = state.block_k_subradii[:, kv_head_idx, :].to(device=query_tensor.device, dtype=torch.float32)
        value_norm = state.block_v_norm_max[:, kv_head_idx].to(device=query_tensor.device, dtype=torch.float32)
        comp_error = state.block_k_comp_error[:, kv_head_idx].to(device=query_tensor.device, dtype=torch.float32)
        center_sim = torch.matmul(center, query_tensor[q_head_idx]) * float(query_scale)
        upper = center_sim + query_norm[q_head_idx] * radius * abs(float(query_scale)) + comp_error
        if int(subcenters.shape[1]) > 1:
            subcenter_sim = torch.einsum("bkd,d->bk", subcenters, query_tensor[q_head_idx]) * float(query_scale)
            sub_upper = subcenter_sim + query_norm[q_head_idx] * subradii * abs(float(query_scale)) + comp_error[:, None]
            upper = torch.minimum(upper, sub_upper.max(dim=1).values)
        normalized_value_norm = value_norm / value_norm.max().clamp_min(1e-6)
        priority = center_sim
        priority = priority + prev_weight * state.block_prev_attention_ema.to(device=query_tensor.device, dtype=torch.float32)
        priority = priority + recency_weight * local_recency
        priority = priority + value_weight * normalized_value_norm
        priority_scores = torch.maximum(priority_scores, priority)
        upper_bounds = torch.maximum(upper_bounds, upper)
    probe_refine_top_k = max(int(getattr(config, "full_attention_probe_refine_top_k", 0)), 0)
    probe_sample_count = max(int(getattr(config, "full_attention_probe_sample_count", 0)), 0)
    if probe_refine_top_k > 0 and probe_sample_count > 0 and num_blocks > 0:
        probe_ranked_block_ids = sorted(
            range(num_blocks),
            key=lambda block_id: float(upper_bounds[int(block_id)].item()),
            reverse=True,
        )[: min(probe_refine_top_k, num_blocks)]
        for block_id in probe_ranked_block_ids:
            probe_upper = _sample_probe_refined_upper_bound(
                state=state,
                block_id=int(block_id),
                query_tensor=query_tensor,
                query_norm=query_norm,
                q_to_kv=q_to_kv,
                query_scale=float(query_scale),
                sample_count=probe_sample_count,
            )
            if probe_upper is not None and math.isfinite(probe_upper):
                upper_bounds[int(block_id)] = min(float(upper_bounds[int(block_id)].item()), float(probe_upper))
    refine_top_k = max(int(getattr(config, "full_attention_refine_top_k", 0)), 0)
    if refine_top_k > 0 and num_blocks > 0:
        ranked_block_ids = sorted(
            range(num_blocks),
            key=lambda block_id: float(upper_bounds[int(block_id)].item()),
            reverse=True,
        )[: min(refine_top_k, num_blocks)]
        for block_id in ranked_block_ids:
            token_start = int(state.block_token_starts[int(block_id)])
            token_count = int(state.block_token_counts[int(block_id)])
            if token_count <= 0:
                continue
            exact_max = float("-inf")
            for q_head_idx in range(int(query_tensor.shape[0])):
                kv_head_idx = int(q_to_kv[q_head_idx])
                key_slice = state.key_cache[kv_head_idx, token_start : token_start + token_count, :].to(
                    device=query_tensor.device,
                    dtype=torch.float32,
                )
                if int(key_slice.shape[0]) == 0:
                    continue
                logits = torch.matmul(key_slice, query_tensor[q_head_idx]) * float(query_scale)
                exact_max = max(exact_max, float(logits.max().item()))
            if math.isfinite(exact_max):
                upper_bounds[int(block_id)] = min(float(upper_bounds[int(block_id)].item()), float(exact_max))
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


def _policy_preference_bonus(
    *,
    score_tensor: Any,
    candidate_block_ids: list[int],
    preferred_block_ids: set[int],
    bias_weight: float,
) -> float:
    if float(bias_weight) <= 0.0 or not candidate_block_ids or not preferred_block_ids:
        return 0.0
    score_values = np.asarray(
        [float(score_tensor[int(block_id)].item()) for block_id in candidate_block_ids],
        dtype=np.float32,
    )
    if score_values.size == 0:
        return 0.0
    scale = float(np.std(score_values))
    if scale < 1e-6:
        scale = max(float(np.max(np.abs(score_values))), 1.0)
    return max(float(bias_weight) * scale, 1e-6)


def _select_diverse_block_ids(
    *,
    ranked_candidate_ids: list[int],
    primary_scores: Any,
    secondary_scores: Any,
    count: int,
    seed_block_ids: list[int],
    diversity_weight: float,
    diversity_radius: int,
    preferred_block_ids: set[int] | None = None,
    preferred_bias_weight: float = 0.0,
) -> list[int]:
    if count <= 0 or not ranked_candidate_ids:
        return []
    if float(diversity_weight) <= 0.0 or int(diversity_radius) <= 0:
        return [int(block_id) for block_id in ranked_candidate_ids[:count]]
    selected: list[int] = []
    selected_anchor_ids = [int(block_id) for block_id in seed_block_ids]
    remaining = [int(block_id) for block_id in ranked_candidate_ids]
    radius = max(int(diversity_radius), 1)
    preferred_ids = {int(block_id) for block_id in (preferred_block_ids or set())}
    preferred_bonus = _policy_preference_bonus(
        score_tensor=primary_scores,
        candidate_block_ids=remaining,
        preferred_block_ids=preferred_ids,
        bias_weight=float(preferred_bias_weight),
    )

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
                + (
                    float(preferred_bonus)
                    if int(block_id) in preferred_ids
                    else 0.0
                )
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
    far_anchor_quota: int,
    far_anchor_priority_margin: float,
    far_anchor_upper_bound_margin: float,
    far_quota: int,
    mid_quota: int,
    near_quota: int,
    seed_block_ids: list[int],
    diversity_weight: float,
    diversity_radius: int,
    preferred_block_ids: set[int] | None = None,
    preferred_bias_weight: float = 0.0,
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
    selected: list[int] = []
    selected_set: set[int] = set()
    if bool(use_upper_bounds_first):
        selected.extend(
            _select_diverse_block_ids(
                ranked_candidate_ids=ranked_by_upper,
                primary_scores=upper_bounds,
                secondary_scores=priority_scores,
                count=max(int(top_k) - len(selected), 0),
                seed_block_ids=seed_block_ids + selected,
                diversity_weight=float(diversity_weight),
                diversity_radius=int(diversity_radius),
                preferred_block_ids=preferred_block_ids,
                preferred_bias_weight=float(preferred_bias_weight),
            )
        )
    else:
        reserved = max(0, min(int(upper_bound_quota), max(int(top_k) - len(selected), 0), len(ranked_by_upper)))
        upper_selected = _select_diverse_block_ids(
            ranked_candidate_ids=ranked_by_upper,
            primary_scores=upper_bounds,
            secondary_scores=priority_scores,
            count=reserved,
            seed_block_ids=seed_block_ids + selected,
            diversity_weight=float(diversity_weight),
            diversity_radius=int(diversity_radius),
            preferred_block_ids=preferred_block_ids,
            preferred_bias_weight=float(preferred_bias_weight),
        )
        selected.extend(int(block_id) for block_id in upper_selected)
        selected_set.update(int(block_id) for block_id in upper_selected)
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
                preferred_block_ids=preferred_block_ids,
                preferred_bias_weight=float(preferred_bias_weight),
            )
            for block_id in region_selected:
                selected.append(block_id)
                selected_set.add(block_id)
            if len(selected) >= int(top_k):
                break
        spill_ranked = [int(block_id) for block_id in ranked_by_priority if int(block_id) not in selected_set]
        spill_selected = _select_diverse_block_ids(
            ranked_candidate_ids=spill_ranked,
            primary_scores=priority_scores,
            secondary_scores=upper_bounds,
            count=max(int(top_k) - len(selected), 0),
            seed_block_ids=seed_block_ids + selected,
            diversity_weight=float(diversity_weight),
            diversity_radius=int(diversity_radius),
            preferred_block_ids=preferred_block_ids,
            preferred_bias_weight=float(preferred_bias_weight),
        )
        for block_id in spill_selected:
            selected.append(int(block_id))
            selected_set.add(int(block_id))
    selected = selected[: int(top_k)]
    if int(far_anchor_quota) <= 0 or not selected:
        return selected
    far_candidates = [
        int(block_id)
        for block_id in ranked_by_priority
        if int(block_id) not in set(selected) and int(region_ids[int(block_id)]) == 0
    ]
    if not far_candidates:
        return selected
    max_replacements = min(int(far_anchor_quota), len(far_candidates), len(selected))
    replacements = 0
    selected_set = set(int(block_id) for block_id in selected)
    for candidate_id in far_candidates:
        if replacements >= max_replacements:
            break
        preferred_weak_ids = [
            int(block_id)
            for block_id in selected
            if int(region_ids[int(block_id)]) != 0
        ]
        weak_pool = preferred_weak_ids if preferred_weak_ids else [int(block_id) for block_id in selected]
        weakest_selected_id = min(
            weak_pool,
            key=lambda block_id: (
                float(priority_scores[int(block_id)].item()),
                float(upper_bounds[int(block_id)].item()),
            ),
        )
        priority_gain = float(priority_scores[int(candidate_id)].item()) - float(
            priority_scores[int(weakest_selected_id)].item()
        )
        upper_gain = float(upper_bounds[int(candidate_id)].item()) - float(
            upper_bounds[int(weakest_selected_id)].item()
        )
        if (
            priority_gain < float(far_anchor_priority_margin)
            and upper_gain < float(far_anchor_upper_bound_margin)
        ):
            continue
        replace_index = selected.index(int(weakest_selected_id))
        selected[replace_index] = int(candidate_id)
        selected_set.remove(int(weakest_selected_id))
        selected_set.add(int(candidate_id))
        replacements += 1
    return selected


def _approximate_preferred_optional_block_ids(
    *,
    candidate_block_ids: list[int],
    region_ids: Any,
    priority_scores: Any,
    upper_bounds: Any,
    top_k: int,
    upper_bound_quota: int,
    far_quota: int,
    mid_quota: int,
    near_quota: int,
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
    selected: list[int] = []
    selected_set: set[int] = set()

    reserved = max(0, min(int(upper_bound_quota), int(top_k), len(ranked_by_upper)))
    for block_id in ranked_by_upper[:reserved]:
        selected.append(int(block_id))
        selected_set.add(int(block_id))
        if len(selected) >= int(top_k):
            return selected[: int(top_k)]

    region_quotas = {
        0: max(0, int(far_quota)),
        1: max(0, int(mid_quota)),
        2: max(0, int(near_quota)),
    }
    for region_id in (0, 1, 2):
        quota = min(region_quotas[region_id], max(int(top_k) - len(selected), 0))
        if quota <= 0:
            continue
        taken = 0
        for block_id in ranked_by_priority:
            if int(block_id) in selected_set or int(region_ids[int(block_id)]) != int(region_id):
                continue
            selected.append(int(block_id))
            selected_set.add(int(block_id))
            taken += 1
            if taken >= quota or len(selected) >= int(top_k):
                break
        if len(selected) >= int(top_k):
            return selected[: int(top_k)]

    for block_id in ranked_by_priority:
        if int(block_id) in selected_set:
            continue
        selected.append(int(block_id))
        if len(selected) >= int(top_k):
            break
    return selected[: int(top_k)]


def _resolve_policy_bias_preferred_optional_ids(
    *,
    state: PersistentFullAttentionLayerState,
    resolved_config: PersistentServingConfig,
    policy_choice: dict[str, Any] | None,
    priority_scores: Any,
    upper_bounds: Any,
) -> tuple[set[int], float]:
    if policy_choice is None or not policy_choice.get("config_overrides"):
        return set(), 0.0
    if float(policy_choice.get("chosen_safe_rate", 0.0)) < float(
        resolved_config.full_attention_shortlist_policy_min_safe_rate
    ):
        return set(), 0.0
    if float(policy_choice.get("matched_oracle_rate", 0.0)) < float(
        resolved_config.full_attention_shortlist_policy_min_matched_oracle_rate
    ):
        return set(), 0.0
    if int(policy_choice.get("vote_count", 0)) < int(resolved_config.full_attention_shortlist_policy_min_vote_count):
        return set(), 0.0
    policy_config = replace(resolved_config, **dict(policy_choice.get("config_overrides", {})))
    num_blocks = int(len(state.block_token_starts))
    recent_ids, mandatory_recent_ids = _resolve_recent_policy(
        num_blocks=num_blocks,
        recent_blocks=int(policy_config.full_attention_recent_block_count),
        mandatory_recent_blocks=policy_config.full_attention_mandatory_recent_block_count,
    )
    mandatory_ids = _mandatory_block_ids(
        num_blocks=num_blocks,
        sink_blocks=int(policy_config.full_attention_sink_block_count),
        mandatory_recent_blocks=mandatory_recent_ids,
    )
    selected_ids: set[int] = set(mandatory_ids)
    soft_recent_ids = [block_id for block_id in recent_ids if block_id not in selected_ids]
    remaining_ids = [block_id for block_id in range(num_blocks) if block_id not in selected_ids]
    exploration_ids = _exploration_block_ids(
        candidate_block_ids=remaining_ids,
        priority_scores=priority_scores,
        per_region=int(policy_config.full_attention_exploration_blocks_per_region),
    )
    selected_ids.update(exploration_ids)
    if not bool(policy_config.enable_priority) or int(policy_config.full_attention_optional_top_k) <= 0:
        return set(), 0.0
    soft_recent_set = set(soft_recent_ids)
    optional_candidates = [block_id for block_id in soft_recent_ids if block_id not in selected_ids]
    optional_candidates.extend(
        block_id
        for block_id in remaining_ids
        if block_id not in selected_ids and block_id not in soft_recent_set
    )
    preferred_ids = _approximate_preferred_optional_block_ids(
        candidate_block_ids=optional_candidates,
        region_ids=state.block_region_ids,
        priority_scores=priority_scores,
        upper_bounds=upper_bounds,
        top_k=int(policy_config.full_attention_optional_top_k),
        upper_bound_quota=int(policy_config.full_attention_optional_upper_bound_quota),
        far_quota=int(policy_config.full_attention_optional_far_quota),
        mid_quota=int(policy_config.full_attention_optional_mid_quota),
        near_quota=int(policy_config.full_attention_optional_near_quota),
    )
    confidence = max(
        float(policy_choice.get("chosen_safe_rate", 0.0)),
        float(policy_choice.get("matched_oracle_rate", 0.0)),
        0.0,
    )
    bias_weight = float(resolved_config.full_attention_shortlist_policy_bias_weight) * max(confidence, 0.25)
    return {int(block_id) for block_id in preferred_ids}, float(bias_weight)


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
            (
                block_k_center,
                block_k_radius,
                block_k_subcenters,
                block_k_subradii,
                block_v_center,
                block_v_radius,
                block_v_norm_max,
                block_prev_attention_ema,
                block_k_comp_error,
            ) = (
                _allocate_full_attention_block_metadata(
                    key_cache=kv_keys,
                    value_cache=kv_values,
                    num_blocks=num_blocks,
                    device=resolved_device,
                    key_centroid_count=int(config.full_attention_key_centroid_count),
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
                block_k_subcenters=block_k_subcenters,
                block_k_subradii=block_k_subradii,
                block_v_center=block_v_center,
                block_v_radius=block_v_radius,
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
                state.block_k_subcenters,
                state.block_k_subradii,
                state.block_v_center,
                state.block_v_radius,
                state.block_v_norm_max,
                state.block_prev_attention_ema,
                state.block_k_comp_error,
            ) = _allocate_full_attention_block_metadata(
                key_cache=state.key_cache,
                value_cache=state.value_cache,
                num_blocks=new_num_blocks,
                device=self.device,
                key_centroid_count=int(self.config.full_attention_key_centroid_count),
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

    def score_blocks(
        self,
        layer_id: int,
        query: Any,
        *,
        query_scale: float,
        config_override: PersistentServingConfig | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        state = self.layers[int(layer_id)]
        resolved_config = config_override or self.config
        priority_scores, upper_bounds = _resolve_block_score_inputs(
            state=state,
            config=resolved_config,
            query=query,
            q_head_to_kv_head=self.q_head_to_kv_head,
            query_scale=float(query_scale),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.telemetry.require_layer(int(layer_id)).score_ms_total += elapsed_ms
        return {
            "priority_scores": priority_scores,
            "upper_bounds": upper_bounds,
        }

    def select_blocks(
        self,
        layer_id: int,
        query: Any,
        *,
        query_scale: float,
        config_override: PersistentServingConfig | None = None,
        policy_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection_start = time.perf_counter()
        state = self.layers[int(layer_id)]
        resolved_config = config_override or self.config
        num_blocks = int(len(state.block_token_starts))
        score_result = self.score_blocks(
            layer_id,
            query,
            query_scale=query_scale,
            config_override=resolved_config,
        )
        priority_scores = score_result["priority_scores"]
        upper_bounds = score_result["upper_bounds"]
        recent_ids, mandatory_recent_ids = _resolve_recent_policy(
            num_blocks=num_blocks,
            recent_blocks=int(resolved_config.full_attention_recent_block_count),
            mandatory_recent_blocks=resolved_config.full_attention_mandatory_recent_block_count,
        )
        mandatory_ids = _mandatory_block_ids(
            num_blocks=num_blocks,
            sink_blocks=int(resolved_config.full_attention_sink_block_count),
            mandatory_recent_blocks=mandatory_recent_ids,
        )
        selected_ids: set[int] = set(mandatory_ids)
        soft_recent_ids = [block_id for block_id in recent_ids if block_id not in selected_ids]
        remaining_ids = [block_id for block_id in range(num_blocks) if block_id not in selected_ids]
        exploration_ids = _exploration_block_ids(
            candidate_block_ids=remaining_ids,
            priority_scores=priority_scores,
            per_region=int(resolved_config.full_attention_exploration_blocks_per_region),
        )
        selected_ids.update(exploration_ids)
        optional_ids: list[int] = []
        policy_preferred_optional_ids: set[int] = set()
        policy_preferred_bias_weight = 0.0
        if str(resolved_config.full_attention_shortlist_policy_mode or "replace").strip().lower() == "bias":
            policy_bias_start = time.perf_counter()
            (
                policy_preferred_optional_ids,
                policy_preferred_bias_weight,
            ) = _resolve_policy_bias_preferred_optional_ids(
                state=state,
                resolved_config=resolved_config,
                policy_choice=policy_choice,
                priority_scores=priority_scores,
                upper_bounds=upper_bounds,
            )
            self.telemetry.require_layer(int(layer_id)).policy_bias_ms_total += (
                (time.perf_counter() - policy_bias_start) * 1000.0
            )
        if bool(resolved_config.enable_priority) and int(resolved_config.full_attention_optional_top_k) > 0:
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
                top_k=int(resolved_config.full_attention_optional_top_k),
                use_upper_bounds_first=bool(resolved_config.full_attention_optional_use_upper_bounds_first),
                upper_bound_quota=int(resolved_config.full_attention_optional_upper_bound_quota),
                far_anchor_quota=int(resolved_config.full_attention_optional_far_anchor_quota),
                far_anchor_priority_margin=float(resolved_config.full_attention_optional_far_anchor_priority_margin),
                far_anchor_upper_bound_margin=float(
                    resolved_config.full_attention_optional_far_anchor_upper_bound_margin
                ),
                far_quota=int(resolved_config.full_attention_optional_far_quota),
                mid_quota=int(resolved_config.full_attention_optional_mid_quota),
                near_quota=int(resolved_config.full_attention_optional_near_quota),
                seed_block_ids=sorted(selected_ids),
                diversity_weight=float(resolved_config.full_attention_optional_diversity_weight),
                diversity_radius=int(resolved_config.full_attention_optional_diversity_radius),
                preferred_block_ids=policy_preferred_optional_ids,
                preferred_bias_weight=float(policy_preferred_bias_weight),
            )
            selected_ids.update(optional_ids)
        else:
            optional_ids = [block_id for block_id in remaining_ids if block_id not in selected_ids]
            selected_ids.update(optional_ids)
        processing_block_ids: list[int] = []
        seen_processing_ids: set[int] = set()
        for block_group in (mandatory_ids, exploration_ids, optional_ids):
            for block_id in block_group:
                if int(block_id) in seen_processing_ids:
                    continue
                processing_block_ids.append(int(block_id))
                seen_processing_ids.add(int(block_id))
        selected_block_ids = sorted(selected_ids)
        self.telemetry.require_layer(int(layer_id)).selection_ms_total += (
            (time.perf_counter() - selection_start) * 1000.0
        )
        return {
            "selected_block_ids": selected_block_ids,
            "processing_block_ids": processing_block_ids,
            "mandatory_block_ids": mandatory_ids,
            "soft_recent_block_ids": soft_recent_ids,
            "exploration_block_ids": exploration_ids,
            "optional_block_ids": optional_ids,
            "priority_scores": priority_scores,
            "upper_bounds": upper_bounds,
            "policy_preferred_optional_block_ids": sorted(int(block_id) for block_id in policy_preferred_optional_ids),
            "policy_preferred_bias_weight": float(policy_preferred_bias_weight),
        }

    def gather_selected_blocks(self, layer_id: int, block_ids: list[int]):
        state = self.layers[int(layer_id)]
        return _gather_selected_block_tensors(state=state, block_ids=block_ids)

    def certify_selected_blocks(
        self,
        layer_id: int,
        *,
        query: Any,
        query_scale: float,
        selected_block_ids: list[int],
        upper_bounds: Any,
        config_override: PersistentServingConfig | None = None,
    ) -> dict[str, Any]:
        state = self.layers[int(layer_id)]
        return _certify_selected_block_frontier(
            state=state,
            query=query,
            q_head_to_kv_head=self.q_head_to_kv_head,
            query_scale=float(query_scale),
            selected_block_ids=selected_block_ids,
            upper_bounds=upper_bounds,
            config=config_override or self.config,
        )

    def stream_decode_layer(
        self,
        layer_id: int,
        query: Any,
        *,
        query_scale: float,
        check_interval: int | None = None,
        stop_on_certificate: bool = False,
        policy_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        torch = _load_torch()
        state = self.layers[int(layer_id)]
        selection = self.select_blocks(
            layer_id,
            query,
            query_scale=query_scale,
            policy_choice=policy_choice,
        )
        priority_scores = selection["priority_scores"]
        upper_bounds = selection["upper_bounds"]
        selected_processing_ids = [int(block_id) for block_id in selection.get("processing_block_ids", [])]
        selected_seen = set(selected_processing_ids)
        remainder_candidates = [
            int(block_id)
            for block_id in range(int(len(state.block_token_starts)))
            if int(block_id) not in selected_seen
        ]
        remainder_ranked = _rank_optional_block_ids(
            candidate_block_ids=remainder_candidates,
            priority_scores=priority_scores,
            upper_bounds=upper_bounds,
            use_upper_bounds_first=bool(self.config.full_attention_optional_use_upper_bounds_first),
        )
        processing_order = [*selected_processing_ids, *remainder_ranked]
        query_tensor = query.to(dtype=torch.float32)
        q_to_kv = np.asarray(self.q_head_to_kv_head, dtype=np.int64)
        num_heads = int(query_tensor.shape[0])
        value_dim = int(state.value_cache.shape[-1])
        m = torch.full((num_heads,), float("-inf"), dtype=torch.float32, device=query_tensor.device)
        l = torch.zeros((num_heads,), dtype=torch.float32, device=query_tensor.device)
        h_accum = torch.zeros((num_heads, value_dim), dtype=torch.float32, device=query_tensor.device)
        metadata_invalid_block_ids = [
            int(block_id)
            for block_id in range(int(len(state.block_token_starts)))
            if float(state.metadata_valid[int(block_id)]) <= 0.0
        ]
        processed_block_ids: list[int] = []
        processed_token_count = 0
        max_bound_excess = 0.0
        checkpoints: list[dict[str, Any]] = []
        first_certified_stop: dict[str, Any] | None = None
        unresolved_block_ids = set(int(block_id) for block_id in processing_order)
        resolved_check_interval = max(
            int(self.config.full_attention_check_interval if check_interval is None else check_interval),
            1,
        )

        for order_index, block_id in enumerate(processing_order, start=1):
            token_start = int(state.block_token_starts[int(block_id)])
            token_count = int(state.block_token_counts[int(block_id)])
            unresolved_block_ids.discard(int(block_id))
            if token_count <= 0:
                continue
            processed_block_ids.append(int(block_id))
            processed_token_count += int(token_count)
            for q_head_idx in range(num_heads):
                kv_head_idx = int(q_to_kv[q_head_idx])
                q_vec = query_tensor[q_head_idx]
                k_slice = state.key_cache[kv_head_idx, token_start : token_start + token_count, :].to(
                    device=query_tensor.device,
                    dtype=torch.float32,
                )
                v_slice = state.value_cache[kv_head_idx, token_start : token_start + token_count, :].to(
                    device=query_tensor.device,
                    dtype=torch.float32,
                )
                logits = torch.matmul(k_slice, q_vec) * float(query_scale)
                if int(logits.numel()) == 0:
                    continue
                block_max = float(logits.max().item())
                max_bound_excess = max(max_bound_excess, block_max - float(upper_bounds[int(block_id)].item()))
                m_old = float(m[q_head_idx].item())
                m_new = float(max(m_old, block_max))
                rescale = math.exp(max(min(m_old - m_new, 0.0), -80.0)) if math.isfinite(m_old) else 0.0
                exp_scores = torch.exp(logits - float(m_new))
                l[q_head_idx] = l[q_head_idx] * float(rescale) + exp_scores.sum()
                h_accum[q_head_idx] = h_accum[q_head_idx] * float(rescale) + torch.sum(
                    exp_scores[:, None] * v_slice,
                    dim=0,
                )
                m[q_head_idx] = float(m_new)

            if order_index % int(resolved_check_interval) != 0 and order_index != len(processing_order):
                continue
            per_head = []
            residual_mass_upper = 0.0
            residual_value_upper = 0.0
            beta_upper = 0.0
            delta_upper = 0.0
            remaining_token_count = int(
                sum(int(state.block_token_counts[int(block_id)]) for block_id in unresolved_block_ids)
            )
            for q_head_idx in range(num_heads):
                kv_head_idx = int(q_to_kv[q_head_idx])
                m_value = float(m[q_head_idx].item())
                l_value = float(l[q_head_idx].item())
                head_residual_mass, head_residual_value = _residual_value_upper_for_blocks(
                    state=state,
                    block_ids=unresolved_block_ids,
                    kv_head_idx=kv_head_idx,
                    q_vec=query_tensor[q_head_idx],
                    q_norm=float(torch.linalg.vector_norm(query_tensor[q_head_idx]).item()),
                    query_scale=float(query_scale),
                    m_value=m_value,
                    upper_bounds=upper_bounds,
                    use_region_caps=bool(self.config.full_attention_region_residual_caps),
                    residual_cluster_count=int(self.config.full_attention_residual_cluster_count),
                )
                denom = float(l_value + head_residual_mass)
                head_beta = float(head_residual_mass / denom) if denom > 0.0 else 0.0
                head_delta = float(head_residual_value / denom) if denom > 0.0 else 0.0
                residual_mass_upper = max(residual_mass_upper, float(head_residual_mass))
                residual_value_upper = max(residual_value_upper, float(head_residual_value))
                beta_upper = max(beta_upper, float(head_beta))
                delta_upper = max(delta_upper, float(head_delta))
                per_head.append(
                    {
                        "q_head_id": int(q_head_idx),
                        "kv_head_id": int(kv_head_idx),
                        "m": float(m_value),
                        "l": float(l_value),
                        "residual_mass_upper": float(head_residual_mass),
                        "residual_value_upper": float(head_residual_value),
                        "beta_upper": float(head_beta),
                        "delta_upper": float(head_delta),
                    }
                )
            instability_reasons: list[str] = []
            if metadata_invalid_block_ids:
                instability_reasons.append("invalid_metadata")
            if float(max_bound_excess) > float(self.config.full_attention_bound_eps):
                instability_reasons.append("bound_exceeded")
            instability_flag = bool(instability_reasons)
            certified_can_stop = (
                int(len(processed_block_ids)) >= max(int(self.config.full_attention_min_processed_blocks), 1)
                and not instability_flag
                and float(beta_upper) < float(self.config.full_attention_mass_eps)
                and float(delta_upper) < float(self.config.full_attention_value_eps)
            )
            checkpoint = {
                "processed_block_count": int(len(processed_block_ids)),
                "processed_token_count": int(processed_token_count),
                "remaining_block_count": int(len(unresolved_block_ids)),
                "remaining_token_count": int(remaining_token_count),
                "beta_upper": float(beta_upper),
                "delta_upper": float(delta_upper),
                "residual_mass_upper": float(residual_mass_upper),
                "residual_value_upper": float(residual_value_upper),
                "max_bound_excess": float(max(0.0, max_bound_excess)),
                "instability_flag": bool(instability_flag),
                "instability_reasons": instability_reasons,
                "certified_can_stop": bool(certified_can_stop),
                "fallback_recommended": bool(
                    instability_flag
                    or (
                        len(unresolved_block_ids) > 0
                        and (
                            float(beta_upper) >= float(self.config.full_attention_mass_eps)
                            or float(delta_upper) >= float(self.config.full_attention_value_eps)
                        )
                    )
                ),
                "per_head": per_head,
            }
            checkpoints.append(checkpoint)
            if first_certified_stop is None and bool(certified_can_stop):
                first_certified_stop = checkpoint
            if bool(stop_on_certificate) and bool(certified_can_stop):
                break
        output = h_accum / l[:, None].clamp_min(1e-8)
        return {
            "output": output,
            "selection": selection,
            "processing_order_block_ids": [int(block_id) for block_id in processing_order],
            "processed_block_ids": [int(block_id) for block_id in processed_block_ids],
            "processed_block_count": int(len(processed_block_ids)),
            "processed_token_count": int(processed_token_count),
            "checkpoint_records": checkpoints,
            "first_certified_stop": first_certified_stop,
            "final_checkpoint": checkpoints[-1] if checkpoints else None,
        }

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
            "persistent_full_attention_score_ms_total_by_layer": {
                str(layer_id): float(self.telemetry.require_layer(layer_id).score_ms_total)
                for layer_id in sorted(self.layers)
            },
            "persistent_full_attention_selection_ms_total_by_layer": {
                str(layer_id): float(self.telemetry.require_layer(layer_id).selection_ms_total)
                for layer_id in sorted(self.layers)
            },
            "persistent_full_attention_policy_bias_ms_total_by_layer": {
                str(layer_id): float(self.telemetry.require_layer(layer_id).policy_bias_ms_total)
                for layer_id in sorted(self.layers)
            },
            "persistent_shortlist_policy_load_ms_total": float(self.telemetry.shortlist_policy_load_ms_total),
            "persistent_shortlist_policy_resolve_ms_total": float(self.telemetry.shortlist_policy_resolve_ms_total),
            "persistent_shortlist_policy_load_count": int(self.telemetry.shortlist_policy_load_count),
            "persistent_shortlist_policy_resolve_count": int(self.telemetry.shortlist_policy_resolve_count),
            "persistent_full_attention_last_beta_upper_by_layer": {
                str(layer_id): (
                    None
                    if state.last_residual_certificate is None
                    else float(state.last_residual_certificate["beta_upper"])
                )
                for layer_id, state in sorted(self.layers.items())
            },
            "persistent_full_attention_last_delta_upper_by_layer": {
                str(layer_id): (
                    None
                    if state.last_residual_certificate is None
                    else float(state.last_residual_certificate["delta_upper"])
                )
                for layer_id, state in sorted(self.layers.items())
            },
            "persistent_full_attention_fallback_recommended_by_layer": {
                str(layer_id): (
                    None
                    if state.last_residual_certificate is None
                    else bool(state.last_residual_certificate["fallback_recommended"])
                )
                for layer_id, state in sorted(self.layers.items())
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

    def decode_full_attention_layer(
        self,
        layer_id: int,
        query: Any,
        *,
        query_scale: float,
        config_override: PersistentServingConfig | None = None,
    ):
        return self.full_attention.decode_layer(layer_id, query, query_scale=query_scale)

    def append_full_attention_step(self, layer_id: int, key_step: Any, value_step: Any, token_index: int) -> None:
        self.full_attention.append_step(layer_id, key_step, value_step, token_index)

    def score_full_attention_blocks(
        self,
        layer_id: int,
        query: Any,
        *,
        query_scale: float,
        config_override: PersistentServingConfig | None = None,
    ) -> dict[str, Any]:
        return self.full_attention.score_blocks(
            layer_id,
            query,
            query_scale=query_scale,
            config_override=config_override,
        )

    def select_full_attention_blocks(
        self,
        layer_id: int,
        query: Any,
        *,
        query_scale: float,
        config_override: PersistentServingConfig | None = None,
        policy_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.full_attention.select_blocks(
            layer_id,
            query,
            query_scale=query_scale,
            config_override=config_override,
            policy_choice=policy_choice,
        )

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

    def certify_full_attention_selected_blocks(
        self,
        layer_id: int,
        *,
        query: Any,
        query_scale: float,
        selected_block_ids: list[int],
        upper_bounds: Any,
        config_override: PersistentServingConfig | None = None,
    ) -> dict[str, Any]:
        return self.full_attention.certify_selected_blocks(
            layer_id,
            query=query,
            query_scale=query_scale,
            selected_block_ids=selected_block_ids,
            upper_bounds=upper_bounds,
            config_override=config_override,
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
