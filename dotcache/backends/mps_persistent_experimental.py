from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np


REGION_SENTINEL = 0
REGION_RECENT = 1
REGION_MID = 2
REGION_FAR = 3

MODE_M0 = 0
MODE_M3 = 3


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in environments without torch
        raise RuntimeError("torch is required for the experimental MPS backend") from exc
    return torch


def _sync_device(device) -> None:
    torch = _load_torch()
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _to_numpy(array_like, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(array_like, dtype=dtype)
    return np.ascontiguousarray(array)


def _host_to_device_nbytes(array_like, *, device) -> int:
    torch = _load_torch()
    if torch.is_tensor(array_like):
        return 0 if array_like.device == device else int(array_like.numel() * array_like.element_size())
    array = np.asarray(array_like)
    return int(array.nbytes)


def _to_device_tensor(array_like, *, device, dtype):
    torch = _load_torch()
    if torch.is_tensor(array_like):
        tensor = array_like.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(np.ascontiguousarray(array_like), dtype=dtype, device=device)
    return tensor.contiguous()


def _resolve_device(device: str | None):
    torch = _load_torch()
    if device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("device=None requires torch_mps to be available for this experimental lane")
    resolved = torch.device(device)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("torch_mps is unavailable on this machine")
    return resolved


def _classify_region(
    *,
    block_token_start: int,
    block_token_end: int,
    total_tokens: int,
    block_size: int,
) -> int:
    if block_token_start < max(block_size, 16):
        return REGION_SENTINEL
    distance = max(total_tokens - block_token_end, 0)
    if distance <= max(block_size * 2, 128):
        return REGION_RECENT
    if distance >= max(total_tokens // 2, block_size * 4):
        return REGION_FAR
    return REGION_MID


def _cost_penalty_from_mode(mode: int) -> float:
    return 1.0 if int(mode) == MODE_M0 else 0.0


def _exploration_prior(region: int) -> float:
    if int(region) == REGION_MID:
        return 0.5
    if int(region) == REGION_FAR:
        return 1.0
    return 0.0


@dataclass(slots=True)
class PageScoreWeights:
    similarity: float = 1.0
    prev_attention: float = 0.5
    distance: float = 0.3
    exploration: float = 0.2
    cost: float = 0.1
    distance_decay: float = 0.001


@dataclass(slots=True)
class PagedAttentionControllerConfig:
    sink_window_tokens: int = 0
    recent_window_tokens: int = 0
    top_k: int = 0
    page_chunk_size: int = 4
    block_size: int = 16
    early_exit: bool = False
    early_exit_eps: float = 1e-4
    mass_eps: float = 1e-4
    value_eps: float = 1e-4
    min_blocks: int = 0
    check_interval: int = 1
    exploration_blocks_per_region: int = 1
    bound_eps: float = 1e-4
    score_weights: PageScoreWeights = field(default_factory=PageScoreWeights)

    def effective_mass_eps(self) -> float:
        return float(self.mass_eps if self.mass_eps > 0.0 else self.early_exit_eps)

    def effective_value_eps(self) -> float:
        return float(self.value_eps if self.value_eps > 0.0 else self.early_exit_eps)


@dataclass(slots=True)
class PagedAttentionSnapshot:
    query: np.ndarray
    page_k_mean: np.ndarray
    prev_attn: np.ndarray
    distance: np.ndarray
    k_pages: np.ndarray
    v_pages: np.ndarray
    page_token_counts: np.ndarray
    page_token_starts: np.ndarray
    source: str = "mps_persistent_experimental"

    @property
    def head_dim(self) -> int:
        return int(self.query.shape[0])

    @property
    def num_pages(self) -> int:
        return int(self.k_pages.shape[0])

    @property
    def tokens_per_page(self) -> int:
        return int(self.k_pages.shape[1])


@dataclass(slots=True)
class ResidentLayerPages:
    device: Any
    query_dtype: Any
    page_k_mean: Any
    prev_attn: Any
    distance: Any
    k_pages: Any
    v_pages: Any
    page_token_counts: Any
    page_token_starts: Any
    page_token_counts_cpu: np.ndarray
    page_token_starts_cpu: np.ndarray
    score_buffer: Any
    acc_buffer: Any
    head_dim: int
    tokens_per_page: int
    num_pages: int
    host_to_device_bytes: int
    block_size: int
    num_blocks: int
    block_page_ids: Any
    block_token_offsets: Any
    block_token_counts: Any
    block_token_starts: Any
    block_distance: Any
    block_regions: Any
    block_modes: Any
    block_metadata_valid: Any
    block_k_center: Any
    block_k_radius: Any
    block_v_norm_max: Any
    block_prev_attention_ema: Any
    block_k_comp_error: Any
    block_upper_buffer: Any
    block_priority_buffer: Any
    block_page_ids_cpu: np.ndarray
    block_token_offsets_cpu: np.ndarray
    block_token_counts_cpu: np.ndarray
    block_token_starts_cpu: np.ndarray
    block_distance_cpu: np.ndarray
    block_regions_cpu: np.ndarray
    block_modes_cpu: np.ndarray
    block_metadata_valid_cpu: np.ndarray
    block_k_radius_cpu: np.ndarray
    block_v_norm_max_cpu: np.ndarray
    block_prev_attention_ema_cpu: np.ndarray
    block_k_comp_error_cpu: np.ndarray

    @property
    def total_tokens(self) -> int:
        if self.num_pages == 0:
            return 0
        return int(self.page_token_starts_cpu[-1] + self.page_token_counts_cpu[-1])


@dataclass(slots=True)
class PageSelectionResult:
    page_scores: np.ndarray
    forced_page_ids: list[int]
    candidate_page_ids: list[int]
    selected_old_page_ids: list[int]
    selected_page_ids: list[int]


@dataclass(slots=True)
class BlockSelectionResult:
    block_upper_bounds: np.ndarray
    block_priorities: np.ndarray
    mandatory_block_ids: list[int]
    exploration_block_ids: list[int]
    optional_block_ids: list[int]
    initial_optional_block_ids: list[int]
    selected_block_ids: list[int]
    selected_page_ids: list[int]


@dataclass(slots=True)
class DecodeResult:
    output: np.ndarray
    processed_page_ids: list[int]
    processed_block_ids: list[int]
    pages_processed: int
    blocks_processed: int
    tokens_processed: int
    early_exit_triggered: bool
    beta_upper: float | None
    delta_upper: float | None
    m_final: float
    l_final: float
    instability_flag: bool
    logits: np.ndarray | None = None
    weights: np.ndarray | None = None


@dataclass(slots=True)
class PagedAttentionStepResult:
    output: np.ndarray
    selected_page_ids: list[int]
    selected_block_ids: list[int]
    processed_page_ids: list[int]
    processed_block_ids: list[int]
    score_time_ms: float
    selection_time_ms: float
    attention_time_ms: float
    total_step_time_ms: float
    selected_page_count: int
    selected_block_count: int
    processed_page_count: int
    processed_block_count: int
    tokens_processed: int
    early_exit_triggered: bool
    beta_upper: float | None
    delta_upper: float | None
    instability_flag: bool
    page_scores: np.ndarray
    block_upper_bounds: np.ndarray
    block_priorities: np.ndarray


def save_paged_attention_snapshot(path: str | Path, snapshot: PagedAttentionSnapshot) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        source=np.asarray(snapshot.source),
        query=np.asarray(snapshot.query, dtype=np.float32),
        page_k_mean=np.asarray(snapshot.page_k_mean, dtype=np.float32),
        prev_attn=np.asarray(snapshot.prev_attn, dtype=np.float32),
        distance=np.asarray(snapshot.distance, dtype=np.float32),
        k_pages=np.asarray(snapshot.k_pages, dtype=np.float32),
        v_pages=np.asarray(snapshot.v_pages, dtype=np.float32),
        page_token_counts=np.asarray(snapshot.page_token_counts, dtype=np.int64),
        page_token_starts=np.asarray(snapshot.page_token_starts, dtype=np.int64),
    )


def load_paged_attention_snapshot(path: str | Path) -> PagedAttentionSnapshot:
    payload = np.load(Path(path), allow_pickle=False)
    return PagedAttentionSnapshot(
        source=str(payload["source"].item()) if "source" in payload else "mps_persistent_experimental",
        query=_to_numpy(payload["query"]),
        page_k_mean=_to_numpy(payload["page_k_mean"]),
        prev_attn=_to_numpy(payload["prev_attn"]),
        distance=_to_numpy(payload["distance"]),
        k_pages=_to_numpy(payload["k_pages"]),
        v_pages=_to_numpy(payload["v_pages"]),
        page_token_counts=np.asarray(payload["page_token_counts"], dtype=np.int64),
        page_token_starts=np.asarray(payload["page_token_starts"], dtype=np.int64),
    )


def build_synthetic_snapshot(
    *,
    num_pages: int,
    tokens_per_page: int,
    head_dim: int,
    seed: int = 0,
    partial_last_page_tokens: int | None = None,
) -> PagedAttentionSnapshot:
    if num_pages <= 0:
        raise ValueError("num_pages must be positive")
    if tokens_per_page <= 0:
        raise ValueError("tokens_per_page must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    rng = np.random.default_rng(seed)
    k_pages = rng.normal(size=(num_pages, tokens_per_page, head_dim)).astype(np.float32)
    v_pages = rng.normal(size=(num_pages, tokens_per_page, head_dim)).astype(np.float32)
    query = rng.normal(size=(head_dim,)).astype(np.float32)
    prev_attn = np.linspace(1.0, 0.05, num_pages, dtype=np.float32)
    distance = np.linspace(float(num_pages - 1), 0.0, num_pages, dtype=np.float32) * float(tokens_per_page)
    page_token_counts = np.full(num_pages, tokens_per_page, dtype=np.int64)
    if partial_last_page_tokens is not None:
        if partial_last_page_tokens <= 0 or partial_last_page_tokens > tokens_per_page:
            raise ValueError("partial_last_page_tokens must be within (0, tokens_per_page]")
        page_token_counts[-1] = int(partial_last_page_tokens)
        if partial_last_page_tokens < tokens_per_page:
            k_pages[-1, partial_last_page_tokens:, :] = 0.0
            v_pages[-1, partial_last_page_tokens:, :] = 0.0
    page_token_starts = np.zeros(num_pages, dtype=np.int64)
    if num_pages > 1:
        page_token_starts[1:] = np.cumsum(page_token_counts[:-1], dtype=np.int64)
    page_k_mean = np.stack(
        [k_pages[index, : int(page_token_counts[index]), :].mean(axis=0) for index in range(num_pages)],
        axis=0,
    ).astype(np.float32, copy=False)
    return PagedAttentionSnapshot(
        query=query,
        page_k_mean=page_k_mean,
        prev_attn=prev_attn,
        distance=distance,
        k_pages=k_pages,
        v_pages=v_pages,
        page_token_counts=page_token_counts,
        page_token_starts=page_token_starts,
    )


def _build_block_metadata(
    *,
    k_pages: np.ndarray,
    v_pages: np.ndarray,
    prev_attn: np.ndarray,
    page_token_counts: np.ndarray,
    page_token_starts: np.ndarray,
    block_size: int,
) -> dict[str, np.ndarray]:
    total_tokens = int(page_token_starts[-1] + page_token_counts[-1]) if page_token_counts.size else 0
    block_page_ids: list[int] = []
    block_token_offsets: list[int] = []
    block_token_counts_list: list[int] = []
    block_token_starts_list: list[int] = []
    block_distance_list: list[float] = []
    block_regions: list[int] = []
    block_modes: list[int] = []
    block_metadata_valid: list[float] = []
    block_k_center: list[np.ndarray] = []
    block_k_radius: list[float] = []
    block_v_norm_max: list[float] = []
    block_prev_attention_ema: list[float] = []
    block_k_comp_error: list[float] = []

    num_pages = int(k_pages.shape[0])
    for page_id in range(num_pages):
        token_count = int(page_token_counts[page_id])
        page_start = int(page_token_starts[page_id])
        for token_offset in range(0, token_count, block_size):
            block_token_count = min(block_size, token_count - token_offset)
            token_start = page_start + token_offset
            token_end = token_start + block_token_count
            k_block = k_pages[page_id, token_offset : token_offset + block_token_count, :]
            v_block = v_pages[page_id, token_offset : token_offset + block_token_count, :]
            k_center = k_block.mean(axis=0).astype(np.float32, copy=False)
            if block_token_count == 1:
                k_radius = 0.0
            else:
                k_radius = float(np.max(np.linalg.norm(k_block - k_center[None, :], axis=1)))
            v_norm_max = float(np.max(np.linalg.norm(v_block, axis=1)))
            block_page_ids.append(page_id)
            block_token_offsets.append(token_offset)
            block_token_counts_list.append(block_token_count)
            block_token_starts_list.append(token_start)
            block_distance_list.append(float(max(total_tokens - token_end, 0)))
            block_regions.append(
                _classify_region(
                    block_token_start=token_start,
                    block_token_end=token_end,
                    total_tokens=total_tokens,
                    block_size=block_size,
                )
            )
            block_modes.append(MODE_M3)
            block_metadata_valid.append(1.0)
            block_k_center.append(k_center)
            block_k_radius.append(k_radius)
            block_v_norm_max.append(v_norm_max)
            block_prev_attention_ema.append(float(prev_attn[page_id]) * float(block_token_count / token_count))
            block_k_comp_error.append(0.0)

    return {
        "block_page_ids": np.asarray(block_page_ids, dtype=np.int64),
        "block_token_offsets": np.asarray(block_token_offsets, dtype=np.int64),
        "block_token_counts": np.asarray(block_token_counts_list, dtype=np.int64),
        "block_token_starts": np.asarray(block_token_starts_list, dtype=np.int64),
        "block_distance": np.asarray(block_distance_list, dtype=np.float32),
        "block_regions": np.asarray(block_regions, dtype=np.int64),
        "block_modes": np.asarray(block_modes, dtype=np.int64),
        "block_metadata_valid": np.asarray(block_metadata_valid, dtype=np.float32),
        "block_k_center": np.asarray(block_k_center, dtype=np.float32),
        "block_k_radius": np.asarray(block_k_radius, dtype=np.float32),
        "block_v_norm_max": np.asarray(block_v_norm_max, dtype=np.float32),
        "block_prev_attention_ema": np.asarray(block_prev_attention_ema, dtype=np.float32),
        "block_k_comp_error": np.asarray(block_k_comp_error, dtype=np.float32),
    }


def prepare_resident_layer_pages(
    *,
    page_k_mean,
    prev_attn,
    distance,
    k_pages,
    v_pages,
    page_token_counts=None,
    page_token_starts=None,
    device: str | None = "mps",
    dtype: str = "float32",
    block_size: int = 16,
) -> ResidentLayerPages:
    torch = _load_torch()
    resolved_device = _resolve_device(device)
    torch_dtype = getattr(torch, dtype)
    page_k_mean_np = _to_numpy(page_k_mean)
    prev_attn_np = _to_numpy(prev_attn)
    distance_np = _to_numpy(distance)
    k_pages_np = _to_numpy(k_pages)
    v_pages_np = _to_numpy(v_pages)
    if page_k_mean_np.ndim != 2:
        raise ValueError("page_k_mean must have shape [num_pages, head_dim]")
    if k_pages_np.ndim != 3 or v_pages_np.ndim != 3:
        raise ValueError("k_pages and v_pages must have shape [num_pages, tokens_per_page, head_dim]")
    if k_pages_np.shape != v_pages_np.shape:
        raise ValueError("k_pages and v_pages must have identical shapes")
    if page_k_mean_np.shape[0] != k_pages_np.shape[0]:
        raise ValueError("page_k_mean page count must match k_pages")
    if page_k_mean_np.shape[1] != k_pages_np.shape[2]:
        raise ValueError("page_k_mean head_dim must match k_pages")
    num_pages = int(k_pages_np.shape[0])
    tokens_per_page = int(k_pages_np.shape[1])
    head_dim = int(k_pages_np.shape[2])
    if prev_attn_np.shape != (num_pages,) or distance_np.shape != (num_pages,):
        raise ValueError("prev_attn and distance must have shape [num_pages]")

    if page_token_counts is None:
        page_token_counts_np = np.full(num_pages, tokens_per_page, dtype=np.int64)
    else:
        page_token_counts_np = np.asarray(page_token_counts, dtype=np.int64)
    if page_token_counts_np.shape != (num_pages,):
        raise ValueError("page_token_counts must have shape [num_pages]")
    if np.any(page_token_counts_np <= 0) or np.any(page_token_counts_np > tokens_per_page):
        raise ValueError("page_token_counts must be within [1, tokens_per_page]")

    if page_token_starts is None:
        page_token_starts_np = np.zeros(num_pages, dtype=np.int64)
        if num_pages > 1:
            page_token_starts_np[1:] = np.cumsum(page_token_counts_np[:-1], dtype=np.int64)
    else:
        page_token_starts_np = np.asarray(page_token_starts, dtype=np.int64)
    if page_token_starts_np.shape != (num_pages,):
        raise ValueError("page_token_starts must have shape [num_pages]")

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    block_size = int(block_size)
    block_metadata = _build_block_metadata(
        k_pages=k_pages_np,
        v_pages=v_pages_np,
        prev_attn=prev_attn_np,
        page_token_counts=page_token_counts_np,
        page_token_starts=page_token_starts_np,
        block_size=block_size,
    )
    num_blocks = int(block_metadata["block_page_ids"].shape[0])

    host_to_device_bytes = (
        _host_to_device_nbytes(page_k_mean, device=resolved_device)
        + _host_to_device_nbytes(prev_attn, device=resolved_device)
        + _host_to_device_nbytes(distance, device=resolved_device)
        + _host_to_device_nbytes(k_pages, device=resolved_device)
        + _host_to_device_nbytes(v_pages, device=resolved_device)
        + _host_to_device_nbytes(page_token_counts_np, device=resolved_device)
        + _host_to_device_nbytes(page_token_starts_np, device=resolved_device)
    )

    return ResidentLayerPages(
        device=resolved_device,
        query_dtype=torch_dtype,
        page_k_mean=_to_device_tensor(page_k_mean_np, device=resolved_device, dtype=torch.float32),
        prev_attn=_to_device_tensor(prev_attn_np, device=resolved_device, dtype=torch.float32),
        distance=_to_device_tensor(distance_np, device=resolved_device, dtype=torch.float32),
        k_pages=_to_device_tensor(k_pages_np, device=resolved_device, dtype=torch_dtype),
        v_pages=_to_device_tensor(v_pages_np, device=resolved_device, dtype=torch_dtype),
        page_token_counts=_to_device_tensor(page_token_counts_np, device=resolved_device, dtype=torch.int64),
        page_token_starts=_to_device_tensor(page_token_starts_np, device=resolved_device, dtype=torch.int64),
        page_token_counts_cpu=page_token_counts_np.copy(),
        page_token_starts_cpu=page_token_starts_np.copy(),
        score_buffer=torch.empty(num_pages, dtype=torch.float32, device=resolved_device),
        acc_buffer=torch.zeros(head_dim, dtype=torch.float32, device=resolved_device),
        head_dim=head_dim,
        tokens_per_page=tokens_per_page,
        num_pages=num_pages,
        host_to_device_bytes=host_to_device_bytes,
        block_size=block_size,
        num_blocks=num_blocks,
        block_page_ids=_to_device_tensor(block_metadata["block_page_ids"], device=resolved_device, dtype=torch.int64),
        block_token_offsets=_to_device_tensor(block_metadata["block_token_offsets"], device=resolved_device, dtype=torch.int64),
        block_token_counts=_to_device_tensor(block_metadata["block_token_counts"], device=resolved_device, dtype=torch.int64),
        block_token_starts=_to_device_tensor(block_metadata["block_token_starts"], device=resolved_device, dtype=torch.int64),
        block_distance=_to_device_tensor(block_metadata["block_distance"], device=resolved_device, dtype=torch.float32),
        block_regions=_to_device_tensor(block_metadata["block_regions"], device=resolved_device, dtype=torch.int64),
        block_modes=_to_device_tensor(block_metadata["block_modes"], device=resolved_device, dtype=torch.int64),
        block_metadata_valid=_to_device_tensor(block_metadata["block_metadata_valid"], device=resolved_device, dtype=torch.float32),
        block_k_center=_to_device_tensor(block_metadata["block_k_center"], device=resolved_device, dtype=torch.float32),
        block_k_radius=_to_device_tensor(block_metadata["block_k_radius"], device=resolved_device, dtype=torch.float32),
        block_v_norm_max=_to_device_tensor(block_metadata["block_v_norm_max"], device=resolved_device, dtype=torch.float32),
        block_prev_attention_ema=_to_device_tensor(block_metadata["block_prev_attention_ema"], device=resolved_device, dtype=torch.float32),
        block_k_comp_error=_to_device_tensor(block_metadata["block_k_comp_error"], device=resolved_device, dtype=torch.float32),
        block_upper_buffer=torch.empty(num_blocks, dtype=torch.float32, device=resolved_device),
        block_priority_buffer=torch.empty(num_blocks, dtype=torch.float32, device=resolved_device),
        block_page_ids_cpu=block_metadata["block_page_ids"].copy(),
        block_token_offsets_cpu=block_metadata["block_token_offsets"].copy(),
        block_token_counts_cpu=block_metadata["block_token_counts"].copy(),
        block_token_starts_cpu=block_metadata["block_token_starts"].copy(),
        block_distance_cpu=block_metadata["block_distance"].copy(),
        block_regions_cpu=block_metadata["block_regions"].copy(),
        block_modes_cpu=block_metadata["block_modes"].copy(),
        block_metadata_valid_cpu=block_metadata["block_metadata_valid"].copy(),
        block_k_radius_cpu=block_metadata["block_k_radius"].copy(),
        block_v_norm_max_cpu=block_metadata["block_v_norm_max"].copy(),
        block_prev_attention_ema_cpu=block_metadata["block_prev_attention_ema"].copy(),
        block_k_comp_error_cpu=block_metadata["block_k_comp_error"].copy(),
    )


def score_pages_reference(
    query,
    page_k_mean,
    prev_attn,
    distance,
    *,
    score_weights: PageScoreWeights | None = None,
) -> np.ndarray:
    weights = PageScoreWeights() if score_weights is None else score_weights
    query_np = _to_numpy(query)
    page_k_mean_np = _to_numpy(page_k_mean)
    prev_attn_np = _to_numpy(prev_attn)
    distance_np = _to_numpy(distance)
    similarity = page_k_mean_np @ query_np
    distance_term = np.exp(-float(weights.distance_decay) * distance_np).astype(np.float32, copy=False)
    return (
        float(weights.similarity) * similarity
        + float(weights.prev_attention) * prev_attn_np
        + float(weights.distance) * distance_term
    ).astype(np.float32, copy=False)


def score_pages_mps(
    query,
    resident: ResidentLayerPages,
    *,
    score_weights: PageScoreWeights | None = None,
):
    torch = _load_torch()
    weights = PageScoreWeights() if score_weights is None else score_weights
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    if tuple(query_tensor.shape) != (resident.head_dim,):
        raise ValueError("query must have shape [head_dim]")
    with torch.no_grad():
        similarity = resident.page_k_mean @ query_tensor
        distance_term = torch.exp(-float(weights.distance_decay) * resident.distance)
        resident.score_buffer.copy_(
            float(weights.similarity) * similarity
            + float(weights.prev_attention) * resident.prev_attn
            + float(weights.distance) * distance_term
        )
    return resident.score_buffer


def _forced_page_ids(
    page_token_starts: np.ndarray,
    page_token_counts: np.ndarray,
    *,
    sink_window_tokens: int,
    recent_window_tokens: int,
) -> list[int]:
    total_tokens = int(page_token_starts[-1] + page_token_counts[-1]) if page_token_starts.size else 0
    recent_threshold = max(total_tokens - int(recent_window_tokens), 0)
    forced: list[int] = []
    for index, (token_start, token_count) in enumerate(zip(page_token_starts.tolist(), page_token_counts.tolist())):
        include_sink = int(sink_window_tokens) > 0 and int(token_start) < int(sink_window_tokens)
        include_recent = int(recent_window_tokens) > 0 and int(token_start + token_count) > recent_threshold
        if include_sink or include_recent:
            forced.append(index)
    return forced


def _select_page_indices_from_scores(
    page_scores: np.ndarray,
    *,
    page_token_starts: np.ndarray,
    page_token_counts: np.ndarray,
    config: PagedAttentionControllerConfig,
) -> PageSelectionResult:
    scores = np.asarray(page_scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError("page_scores must have shape [num_pages]")
    forced_page_ids = _forced_page_ids(
        page_token_starts,
        page_token_counts,
        sink_window_tokens=config.sink_window_tokens,
        recent_window_tokens=config.recent_window_tokens,
    )
    forced_set = set(forced_page_ids)
    candidate_page_ids = [index for index in range(scores.shape[0]) if index not in forced_set]
    selected_old_page_ids: list[int] = []
    if config.top_k > 0 and candidate_page_ids:
        candidate_scores = scores[np.asarray(candidate_page_ids, dtype=np.int64)]
        order = np.lexsort((np.asarray(candidate_page_ids, dtype=np.int64), -candidate_scores))
        chosen = order[: min(int(config.top_k), len(candidate_page_ids))]
        selected_old_page_ids = [candidate_page_ids[int(index)] for index in chosen.tolist()]
    selected_page_ids = sorted(forced_set.union(selected_old_page_ids))
    return PageSelectionResult(
        page_scores=scores.copy(),
        forced_page_ids=forced_page_ids,
        candidate_page_ids=candidate_page_ids,
        selected_old_page_ids=selected_old_page_ids,
        selected_page_ids=selected_page_ids,
    )


def select_pages_reference(
    query,
    *,
    page_k_mean,
    prev_attn,
    distance,
    page_token_starts,
    page_token_counts,
    config: PagedAttentionControllerConfig,
) -> PageSelectionResult:
    page_scores = score_pages_reference(
        query,
        page_k_mean,
        prev_attn,
        distance,
        score_weights=config.score_weights,
    )
    return _select_page_indices_from_scores(
        page_scores,
        page_token_starts=np.asarray(page_token_starts, dtype=np.int64),
        page_token_counts=np.asarray(page_token_counts, dtype=np.int64),
        config=config,
    )


def select_pages_mps(
    query,
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
) -> PageSelectionResult:
    page_scores = score_pages_mps(query, resident, score_weights=config.score_weights)
    _sync_device(resident.device)
    page_scores_np = page_scores.detach().cpu().numpy()
    return _select_page_indices_from_scores(
        page_scores_np,
        page_token_starts=resident.page_token_starts_cpu,
        page_token_counts=resident.page_token_counts_cpu,
        config=config,
    )


def _block_ids_for_page_ids(resident: ResidentLayerPages, page_ids: list[int]) -> list[int]:
    page_set = set(int(page_id) for page_id in page_ids)
    block_ids = [
        int(block_id)
        for block_id, page_id in enumerate(resident.block_page_ids_cpu.tolist())
        if int(page_id) in page_set
    ]
    block_ids.sort(key=lambda block_id: int(resident.block_token_starts_cpu[block_id]))
    return block_ids


def _stack_selected_blocks_torch(
    resident: ResidentLayerPages,
    selected_block_ids: list[int],
):
    torch = _load_torch()
    k_chunks = []
    v_chunks = []
    for block_id in selected_block_ids:
        page_id = int(resident.block_page_ids_cpu[block_id])
        token_offset = int(resident.block_token_offsets_cpu[block_id])
        token_count = int(resident.block_token_counts_cpu[block_id])
        k_chunks.append(resident.k_pages[page_id, token_offset : token_offset + token_count, :])
        v_chunks.append(resident.v_pages[page_id, token_offset : token_offset + token_count, :])
    if not k_chunks:
        raise ValueError("selected_block_ids must be non-empty")
    return torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)


def _stack_selected_blocks_numpy(
    *,
    k_pages: np.ndarray,
    v_pages: np.ndarray,
    block_page_ids: np.ndarray,
    block_token_offsets: np.ndarray,
    block_token_counts: np.ndarray,
    selected_block_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not selected_block_ids:
        raise ValueError("selected_block_ids must be non-empty")
    flat_k = np.concatenate(
        [
            k_pages[
                int(block_page_ids[block_id]),
                int(block_token_offsets[block_id]) : int(block_token_offsets[block_id] + block_token_counts[block_id]),
                :,
            ]
            for block_id in selected_block_ids
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    flat_v = np.concatenate(
        [
            v_pages[
                int(block_page_ids[block_id]),
                int(block_token_offsets[block_id]) : int(block_token_offsets[block_id] + block_token_counts[block_id]),
                :,
            ]
            for block_id in selected_block_ids
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return flat_k, flat_v


def _page_ids_from_block_ids(resident: ResidentLayerPages, block_ids: list[int]) -> list[int]:
    return sorted({int(resident.block_page_ids_cpu[block_id]) for block_id in block_ids})


def _compute_block_bounds_and_priorities_reference(
    query,
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    q = _to_numpy(query)
    q_norm = float(np.linalg.norm(q))
    scale = 1.0 / math.sqrt(float(resident.head_dim))
    similarity = resident.block_k_center.detach().cpu().numpy() @ q
    upper_bounds = (
        similarity * scale
        + q_norm * (resident.block_k_radius_cpu + resident.block_k_comp_error_cpu) * scale
    ).astype(np.float32, copy=False)
    recency_bonus = np.exp(-float(config.score_weights.distance_decay) * resident.block_distance_cpu).astype(np.float32, copy=False)
    exploration_bonus = np.asarray(
        [_exploration_prior(int(region)) for region in resident.block_regions_cpu.tolist()],
        dtype=np.float32,
    )
    cost_penalty = np.asarray(
        [_cost_penalty_from_mode(int(mode)) for mode in resident.block_modes_cpu.tolist()],
        dtype=np.float32,
    )
    priorities = (
        float(config.score_weights.similarity) * upper_bounds
        + float(config.score_weights.prev_attention) * resident.block_prev_attention_ema_cpu
        + float(config.score_weights.distance) * recency_bonus
        + float(config.score_weights.exploration) * exploration_bonus
        - float(config.score_weights.cost) * cost_penalty
    ).astype(np.float32, copy=False)
    return upper_bounds, priorities


def _compute_block_bounds_and_priorities_mps(
    query,
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _load_torch()
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    q_norm = torch.linalg.vector_norm(query_tensor)
    scale = 1.0 / math.sqrt(float(resident.head_dim))
    with torch.no_grad():
        upper_bounds = (
            torch.matmul(resident.block_k_center, query_tensor) * scale
            + q_norm * (resident.block_k_radius + resident.block_k_comp_error) * scale
        )
        recency_bonus = torch.exp(-float(config.score_weights.distance_decay) * resident.block_distance)
        exploration_bonus = torch.where(
            resident.block_regions == REGION_MID,
            torch.tensor(0.5, dtype=torch.float32, device=resident.device),
            torch.where(
                resident.block_regions == REGION_FAR,
                torch.tensor(1.0, dtype=torch.float32, device=resident.device),
                torch.tensor(0.0, dtype=torch.float32, device=resident.device),
            ),
        )
        cost_penalty = torch.where(
            resident.block_modes == MODE_M0,
            torch.tensor(1.0, dtype=torch.float32, device=resident.device),
            torch.tensor(0.0, dtype=torch.float32, device=resident.device),
        )
        resident.block_upper_buffer.copy_(upper_bounds)
        resident.block_priority_buffer.copy_(
            float(config.score_weights.similarity) * resident.block_upper_buffer
            + float(config.score_weights.prev_attention) * resident.block_prev_attention_ema
            + float(config.score_weights.distance) * recency_bonus
            + float(config.score_weights.exploration) * exploration_bonus
            - float(config.score_weights.cost) * cost_penalty
        )
    _sync_device(resident.device)
    return (
        resident.block_upper_buffer.detach().cpu().numpy().astype(np.float32, copy=False),
        resident.block_priority_buffer.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _mandatory_block_ids(
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
) -> list[int]:
    total_tokens = resident.total_tokens
    recent_threshold = max(total_tokens - int(config.recent_window_tokens), 0)
    mandatory: list[int] = []
    for block_id in range(resident.num_blocks):
        token_start = int(resident.block_token_starts_cpu[block_id])
        token_end = token_start + int(resident.block_token_counts_cpu[block_id])
        include_sink = int(config.sink_window_tokens) > 0 and token_start < int(config.sink_window_tokens)
        include_recent = int(config.recent_window_tokens) > 0 and token_end > recent_threshold
        if include_sink or include_recent:
            mandatory.append(block_id)
    mandatory.sort(key=lambda block_id: int(resident.block_token_starts_cpu[block_id]))
    return mandatory


def _exploration_block_ids(
    block_priorities: np.ndarray,
    resident: ResidentLayerPages,
    *,
    excluded_block_ids: set[int],
    config: PagedAttentionControllerConfig,
) -> list[int]:
    if config.exploration_blocks_per_region <= 0 or resident.num_blocks <= len(excluded_block_ids):
        return []
    exploration_ids: list[int] = []
    ranked_optional = sorted(
        (
            block_id
            for block_id in range(resident.num_blocks)
            if block_id not in excluded_block_ids
        ),
        key=lambda block_id: (-float(block_priorities[block_id]), int(block_id)),
    )
    top_ranked = set(ranked_optional[: max(int(config.top_k), 1)])
    for region in (REGION_MID, REGION_FAR):
        region_candidates = [
            block_id
            for block_id in ranked_optional
            if int(resident.block_regions_cpu[block_id]) == region and block_id not in exploration_ids
        ]
        preferred = [block_id for block_id in region_candidates if block_id not in top_ranked]
        chosen_pool = preferred if preferred else region_candidates
        for block_id in chosen_pool[: int(config.exploration_blocks_per_region)]:
            exploration_ids.append(int(block_id))
    exploration_ids.sort(key=lambda block_id: int(resident.block_token_starts_cpu[block_id]))
    return exploration_ids


def _select_blocks_from_bounds(
    block_upper_bounds: np.ndarray,
    block_priorities: np.ndarray,
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
) -> BlockSelectionResult:
    mandatory_block_ids = _mandatory_block_ids(resident, config=config)
    excluded = set(mandatory_block_ids)
    exploration_block_ids = _exploration_block_ids(
        block_priorities,
        resident,
        excluded_block_ids=excluded,
        config=config,
    )
    excluded.update(exploration_block_ids)
    optional_block_ids = [
        block_id
        for block_id in range(resident.num_blocks)
        if block_id not in excluded
    ]
    optional_block_ids.sort(key=lambda block_id: (-float(block_priorities[block_id]), int(block_id)))
    if config.top_k > 0:
        initial_optional_block_ids = optional_block_ids[: min(int(config.top_k), len(optional_block_ids))]
    else:
        initial_optional_block_ids = list(optional_block_ids)
    selected_block_ids = sorted(
        set(mandatory_block_ids).union(exploration_block_ids).union(initial_optional_block_ids),
        key=lambda block_id: int(resident.block_token_starts_cpu[block_id]),
    )
    selected_page_ids = _page_ids_from_block_ids(resident, selected_block_ids)
    return BlockSelectionResult(
        block_upper_bounds=np.asarray(block_upper_bounds, dtype=np.float32).copy(),
        block_priorities=np.asarray(block_priorities, dtype=np.float32).copy(),
        mandatory_block_ids=mandatory_block_ids,
        exploration_block_ids=exploration_block_ids,
        optional_block_ids=optional_block_ids,
        initial_optional_block_ids=initial_optional_block_ids,
        selected_block_ids=selected_block_ids,
        selected_page_ids=selected_page_ids,
    )


def decode_selected_pages_reference(
    query,
    *,
    k_pages,
    v_pages,
    selected_page_ids: list[int],
    page_token_counts,
    block_size: int = 16,
) -> DecodeResult:
    if not selected_page_ids:
        raise ValueError("selected_page_ids must be non-empty")
    k_pages_np = _to_numpy(k_pages)
    v_pages_np = _to_numpy(v_pages)
    page_token_counts_np = np.asarray(page_token_counts, dtype=np.int64)
    page_token_starts_np = np.zeros_like(page_token_counts_np)
    if page_token_counts_np.size > 1:
        page_token_starts_np[1:] = np.cumsum(page_token_counts_np[:-1], dtype=np.int64)
    resident = prepare_resident_layer_pages(
        page_k_mean=np.stack(
            [k_pages_np[index, : int(page_token_counts_np[index]), :].mean(axis=0) for index in range(k_pages_np.shape[0])],
            axis=0,
        ),
        prev_attn=np.ones(k_pages_np.shape[0], dtype=np.float32),
        distance=np.zeros(k_pages_np.shape[0], dtype=np.float32),
        k_pages=k_pages_np,
        v_pages=v_pages_np,
        page_token_counts=page_token_counts_np,
        page_token_starts=page_token_starts_np,
        device="cpu",
        block_size=block_size,
    )
    selected_block_ids = _block_ids_for_page_ids(resident, selected_page_ids)
    flat_k, flat_v = _stack_selected_blocks_numpy(
        k_pages=k_pages_np,
        v_pages=v_pages_np,
        block_page_ids=resident.block_page_ids_cpu,
        block_token_offsets=resident.block_token_offsets_cpu,
        block_token_counts=resident.block_token_counts_cpu,
        selected_block_ids=selected_block_ids,
    )
    query_np = _to_numpy(query)
    logits = flat_k @ query_np
    shifted = logits - np.max(logits)
    weights = np.exp(shifted).astype(np.float32, copy=False)
    weights = weights / np.sum(weights)
    output = weights @ flat_v
    processed_page_ids = sorted(set(int(page_id) for page_id in selected_page_ids))
    return DecodeResult(
        output=output.astype(np.float32, copy=False),
        processed_page_ids=processed_page_ids,
        processed_block_ids=selected_block_ids,
        pages_processed=len(processed_page_ids),
        blocks_processed=len(selected_block_ids),
        tokens_processed=int(logits.shape[0]),
        early_exit_triggered=False,
        beta_upper=None,
        delta_upper=None,
        m_final=float(np.max(logits)),
        l_final=float(np.sum(np.exp(shifted.astype(np.float64)))),
        instability_flag=False,
        logits=logits.astype(np.float32, copy=False),
        weights=weights.astype(np.float32, copy=False),
    )


def decode_selected_pages_dense_mps(
    query,
    resident: ResidentLayerPages,
    selected_page_ids: list[int],
) -> DecodeResult:
    torch = _load_torch()
    if not selected_page_ids:
        raise ValueError("selected_page_ids must be non-empty")
    selected_block_ids = _block_ids_for_page_ids(resident, selected_page_ids)
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    with torch.no_grad():
        flat_k, flat_v = _stack_selected_blocks_torch(resident, selected_block_ids)
        logits = torch.matmul(flat_k.to(dtype=torch.float32), query_tensor)
        weights = torch.softmax(logits, dim=0)
        output = torch.matmul(weights.to(dtype=flat_v.dtype), flat_v).to(dtype=torch.float32)
    processed_page_ids = _page_ids_from_block_ids(resident, selected_block_ids)
    return DecodeResult(
        output=output.detach().cpu().numpy(),
        processed_page_ids=processed_page_ids,
        processed_block_ids=selected_block_ids,
        pages_processed=len(processed_page_ids),
        blocks_processed=len(selected_block_ids),
        tokens_processed=int(logits.numel()),
        early_exit_triggered=False,
        beta_upper=None,
        delta_upper=None,
        m_final=float(torch.max(logits).item()),
        l_final=float(torch.sum(torch.exp(logits - torch.max(logits))).item()),
        instability_flag=False,
        logits=logits.detach().cpu().numpy(),
        weights=weights.detach().cpu().numpy(),
    )


def decode_selected_pages_mps(
    query,
    resident: ResidentLayerPages,
    selected_page_ids: list[int],
    *,
    page_chunk_size: int = 4,
    early_exit: bool = False,
    early_exit_eps: float = 1e-4,
    return_debug: bool = False,
) -> DecodeResult:
    torch = _load_torch()
    if not selected_page_ids:
        raise ValueError("selected_page_ids must be non-empty")
    if page_chunk_size <= 0:
        raise ValueError("page_chunk_size must be positive")
    selected_block_ids = _block_ids_for_page_ids(resident, selected_page_ids)
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    acc = resident.acc_buffer.zero_()
    logits_parts: list[np.ndarray] = []
    processed_block_ids: list[int] = []
    tokens_processed = 0
    m = None
    l = None
    with torch.no_grad():
        for block_start in range(0, len(selected_block_ids), page_chunk_size):
            chunk_block_ids = selected_block_ids[block_start : block_start + page_chunk_size]
            k_chunk, v_chunk = _stack_selected_blocks_torch(resident, chunk_block_ids)
            logits = torch.matmul(k_chunk.to(dtype=torch.float32), query_tensor)
            chunk_max = torch.max(logits)
            if m is None or l is None:
                exp_scores = torch.exp(logits - chunk_max)
                m = chunk_max
                l = torch.sum(exp_scores)
                acc.copy_(torch.sum(exp_scores[:, None] * v_chunk.to(dtype=torch.float32), dim=0))
            else:
                m_new = torch.maximum(m, chunk_max)
                alpha = torch.exp(m - m_new)
                exp_scores = torch.exp(logits - m_new)
                l = l * alpha + torch.sum(exp_scores)
                acc.mul_(alpha).add_(torch.sum(exp_scores[:, None] * v_chunk.to(dtype=torch.float32), dim=0))
                m = m_new
            tokens_processed += int(logits.numel())
            processed_block_ids.extend(chunk_block_ids)
            if return_debug:
                logits_parts.append(logits.detach().cpu().numpy())
            if early_exit and len(processed_block_ids) >= max(1, len(selected_block_ids) - 1):
                break

    if m is None or l is None:
        raise RuntimeError("failed to process any attention blocks")
    output = (acc / l).detach().cpu().numpy().astype(np.float32, copy=False)
    logits_np = None
    weights_np = None
    if return_debug and logits_parts:
        logits_np = np.concatenate(logits_parts, axis=0).astype(np.float32, copy=False)
        shifted = logits_np - np.max(logits_np)
        weights_np = np.exp(shifted).astype(np.float32, copy=False)
        weights_np = (weights_np / np.sum(weights_np)).astype(np.float32, copy=False)
    processed_page_ids = _page_ids_from_block_ids(resident, processed_block_ids)
    return DecodeResult(
        output=output,
        processed_page_ids=processed_page_ids,
        processed_block_ids=processed_block_ids,
        pages_processed=len(processed_page_ids),
        blocks_processed=len(processed_block_ids),
        tokens_processed=tokens_processed,
        early_exit_triggered=False,
        beta_upper=None,
        delta_upper=None,
        m_final=float(m.detach().cpu().item()),
        l_final=float(l.detach().cpu().item()),
        instability_flag=False,
        logits=logits_np,
        weights=weights_np,
    )


def _initialize_remaining_bounds(
    remaining_block_ids: set[int],
    *,
    block_upper_bounds: np.ndarray,
    resident: ResidentLayerPages,
    m_value: float,
) -> tuple[float, float]:
    if not remaining_block_ids:
        return 0.0, 0.0
    remaining_ids = np.asarray(sorted(remaining_block_ids), dtype=np.int64)
    mass_terms = resident.block_token_counts_cpu[remaining_ids].astype(np.float64) * np.exp(
        block_upper_bounds[remaining_ids].astype(np.float64) - float(m_value)
    )
    value_terms = mass_terms * resident.block_v_norm_max_cpu[remaining_ids].astype(np.float64)
    return float(np.sum(mass_terms)), float(np.sum(value_terms))


def run_paged_attention_step(
    query,
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
    engine: Literal["mps_experimental", "torch_mps_baseline"] = "mps_experimental",
) -> PagedAttentionStepResult:
    torch = _load_torch()
    total_start = time.perf_counter()
    score_start = time.perf_counter()
    block_upper_bounds, block_priorities = _compute_block_bounds_and_priorities_mps(query, resident, config=config)
    page_scores = score_pages_mps(query, resident, score_weights=config.score_weights)
    _sync_device(resident.device)
    score_time_ms = (time.perf_counter() - score_start) * 1000.0

    selection_start = time.perf_counter()
    selection = _select_blocks_from_bounds(block_upper_bounds, block_priorities, resident, config=config)
    selection_time_ms = (time.perf_counter() - selection_start) * 1000.0

    attention_start = time.perf_counter()
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    acc = resident.acc_buffer.zero_()
    processed_block_ids: list[int] = []
    processed_page_ids: list[int] = []
    processed_page_set: set[int] = set()
    processed_blocks = 0
    tokens_processed = 0
    m = None
    l = None
    beta_upper = None
    delta_upper = None
    early_exit_triggered = False
    instability_flag = not bool(np.all(resident.block_metadata_valid_cpu > 0.5))
    if resident.num_blocks > 0 and int(config.sink_window_tokens) + int(config.recent_window_tokens) >= resident.total_tokens:
        instability_flag = True

    mandatory_queue = list(selection.mandatory_block_ids)
    exploration_queue = list(selection.exploration_block_ids)
    optional_sorted = list(selection.optional_block_ids)
    frontier_size = len(optional_sorted) if int(config.top_k) <= 0 else min(int(config.top_k), len(optional_sorted))
    frontier_index = 0
    optional_heap: list[tuple[float, int]] = []
    for block_id in optional_sorted[:frontier_size]:
        heapq.heappush(optional_heap, (-float(block_priorities[block_id]), int(block_id)))
    frontier_index = frontier_size
    max_optional_without_fallback = frontier_size

    all_unprocessed: set[int] = set(range(resident.num_blocks))
    remaining_mass = None
    remaining_value = None
    previous_beta = None
    beta_non_decreasing_checks = 0

    def process_block(block_id: int) -> None:
        nonlocal m, l, tokens_processed, processed_blocks, remaining_mass, remaining_value, instability_flag
        page_id = int(resident.block_page_ids_cpu[block_id])
        token_offset = int(resident.block_token_offsets_cpu[block_id])
        token_count = int(resident.block_token_counts_cpu[block_id])
        k_block = resident.k_pages[page_id, token_offset : token_offset + token_count, :].to(dtype=torch.float32)
        v_block = resident.v_pages[page_id, token_offset : token_offset + token_count, :]
        logits = torch.matmul(k_block, query_tensor)
        chunk_max = torch.max(logits)
        if m is None or l is None:
            exp_scores = torch.exp(logits - chunk_max)
            m = chunk_max
            l = torch.sum(exp_scores)
            acc.copy_(torch.sum(exp_scores[:, None] * v_block.to(dtype=torch.float32), dim=0))
            all_unprocessed.discard(block_id)
            remaining_mass, remaining_value = _initialize_remaining_bounds(
                all_unprocessed,
                block_upper_bounds=block_upper_bounds,
                resident=resident,
                m_value=float(m.detach().cpu().item()),
            )
        else:
            old_m = float(m.detach().cpu().item())
            m_new = torch.maximum(m, chunk_max)
            alpha = torch.exp(m - m_new)
            exp_scores = torch.exp(logits - m_new)
            l = l * alpha + torch.sum(exp_scores)
            acc.mul_(alpha).add_(torch.sum(exp_scores[:, None] * v_block.to(dtype=torch.float32), dim=0))
            m = m_new
            new_m = float(m.detach().cpu().item())
            if remaining_mass is not None and remaining_value is not None:
                if new_m > old_m:
                    rescale = math.exp(old_m - new_m)
                    remaining_mass *= rescale
                    remaining_value *= rescale
                term = float(resident.block_token_counts_cpu[block_id]) * math.exp(float(block_upper_bounds[block_id]) - new_m)
                remaining_mass = max(0.0, remaining_mass - term)
                remaining_value = max(0.0, remaining_value - term * float(resident.block_v_norm_max_cpu[block_id]))
            all_unprocessed.discard(block_id)

        if float(torch.max(logits).detach().cpu().item()) > float(block_upper_bounds[block_id]) + float(config.bound_eps):
            instability_flag = True
        processed_blocks += 1
        tokens_processed += int(logits.numel())
        processed_block_ids.append(int(block_id))
        if page_id not in processed_page_set:
            processed_page_set.add(page_id)
            processed_page_ids.append(page_id)

    while mandatory_queue:
        process_block(mandatory_queue.pop(0))

    while exploration_queue:
        process_block(exploration_queue.pop(0))

    mandatory_done = True
    effective_min_blocks = int(config.min_blocks) if int(config.min_blocks) > 0 else len(selection.mandatory_block_ids) + 2
    optional_processed = 0

    while optional_heap or frontier_index < len(optional_sorted):
        if not optional_heap:
            widen = len(optional_sorted) if not config.early_exit or int(config.top_k) <= 0 else min(int(config.top_k), len(optional_sorted) - frontier_index)
            for block_id in optional_sorted[frontier_index : frontier_index + widen]:
                heapq.heappush(optional_heap, (-float(block_priorities[block_id]), int(block_id)))
            frontier_index += widen
            if not optional_heap:
                break
        _priority, block_id = heapq.heappop(optional_heap)
        process_block(block_id)
        optional_processed += 1

        if not config.early_exit and int(config.top_k) > 0 and optional_processed >= max_optional_without_fallback:
            break

        if not config.early_exit:
            continue

        if processed_blocks % max(int(config.check_interval), 1) != 0:
            continue
        if remaining_mass is None or remaining_value is None or l is None:
            continue
        denom = float(l.detach().cpu().item()) + float(remaining_mass)
        if denom <= 0.0:
            continue
        beta_upper = float(remaining_mass / denom)
        delta_upper = float(remaining_value / denom)
        if previous_beta is not None and beta_upper >= previous_beta - 1e-12:
            beta_non_decreasing_checks += 1
        else:
            beta_non_decreasing_checks = 0
        previous_beta = beta_upper
        if beta_non_decreasing_checks >= 2:
            instability_flag = True
        if (
            mandatory_done
            and processed_blocks >= effective_min_blocks
            and beta_upper < config.effective_mass_eps()
            and delta_upper < config.effective_value_eps()
            and not instability_flag
        ):
            early_exit_triggered = True
            break

    if m is None or l is None:
        raise RuntimeError("failed to process any blocks")
    output = (acc / l).detach().cpu().numpy().astype(np.float32, copy=False)
    attention_time_ms = (time.perf_counter() - attention_start) * 1000.0
    total_step_time_ms = (time.perf_counter() - total_start) * 1000.0
    processed_page_ids_sorted = sorted(processed_page_set)
    return PagedAttentionStepResult(
        output=output,
        selected_page_ids=selection.selected_page_ids,
        selected_block_ids=selection.selected_block_ids,
        processed_page_ids=processed_page_ids_sorted,
        processed_block_ids=processed_block_ids,
        score_time_ms=score_time_ms,
        selection_time_ms=selection_time_ms,
        attention_time_ms=attention_time_ms,
        total_step_time_ms=total_step_time_ms,
        selected_page_count=len(selection.selected_page_ids),
        selected_block_count=len(selection.selected_block_ids),
        processed_page_count=len(processed_page_ids_sorted),
        processed_block_count=len(processed_block_ids),
        tokens_processed=tokens_processed,
        early_exit_triggered=early_exit_triggered,
        beta_upper=beta_upper,
        delta_upper=delta_upper,
        instability_flag=instability_flag,
        page_scores=page_scores.detach().cpu().numpy().astype(np.float32, copy=False),
        block_upper_bounds=selection.block_upper_bounds,
        block_priorities=selection.block_priorities,
    )


def run_reference_step(
    snapshot: PagedAttentionSnapshot,
    *,
    config: PagedAttentionControllerConfig,
) -> PagedAttentionStepResult:
    total_start = time.perf_counter()
    resident = prepare_resident_layer_pages(
        page_k_mean=snapshot.page_k_mean,
        prev_attn=snapshot.prev_attn,
        distance=snapshot.distance,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        page_token_counts=snapshot.page_token_counts,
        page_token_starts=snapshot.page_token_starts,
        device="cpu",
        block_size=config.block_size,
    )

    score_start = time.perf_counter()
    block_upper_bounds, block_priorities = _compute_block_bounds_and_priorities_reference(snapshot.query, resident, config=config)
    page_scores = score_pages_reference(
        snapshot.query,
        snapshot.page_k_mean,
        snapshot.prev_attn,
        snapshot.distance,
        score_weights=config.score_weights,
    )
    score_time_ms = (time.perf_counter() - score_start) * 1000.0

    selection_start = time.perf_counter()
    selection = _select_blocks_from_bounds(block_upper_bounds, block_priorities, resident, config=config)
    selection_time_ms = (time.perf_counter() - selection_start) * 1000.0

    attention_start = time.perf_counter()
    flat_k, flat_v = _stack_selected_blocks_numpy(
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        block_page_ids=resident.block_page_ids_cpu,
        block_token_offsets=resident.block_token_offsets_cpu,
        block_token_counts=resident.block_token_counts_cpu,
        selected_block_ids=selection.selected_block_ids,
    )
    query_np = _to_numpy(snapshot.query)
    logits = flat_k @ query_np
    shifted = logits - np.max(logits)
    weights = np.exp(shifted).astype(np.float32, copy=False)
    weights = weights / np.sum(weights)
    output = weights @ flat_v
    attention_time_ms = (time.perf_counter() - attention_start) * 1000.0

    total_step_time_ms = (time.perf_counter() - total_start) * 1000.0
    processed_page_ids = _page_ids_from_block_ids(resident, selection.selected_block_ids)
    return PagedAttentionStepResult(
        output=output.astype(np.float32, copy=False),
        selected_page_ids=selection.selected_page_ids,
        selected_block_ids=selection.selected_block_ids,
        processed_page_ids=processed_page_ids,
        processed_block_ids=selection.selected_block_ids,
        score_time_ms=score_time_ms,
        selection_time_ms=selection_time_ms,
        attention_time_ms=attention_time_ms,
        total_step_time_ms=total_step_time_ms,
        selected_page_count=len(selection.selected_page_ids),
        selected_block_count=len(selection.selected_block_ids),
        processed_page_count=len(processed_page_ids),
        processed_block_count=len(selection.selected_block_ids),
        tokens_processed=int(logits.shape[0]),
        early_exit_triggered=False,
        beta_upper=None,
        delta_upper=None,
        instability_flag=False,
        page_scores=np.asarray(page_scores, dtype=np.float32).copy(),
        block_upper_bounds=selection.block_upper_bounds,
        block_priorities=selection.block_priorities,
    )


def result_error_stats(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=np.float32)
    reference_arr = np.asarray(reference, dtype=np.float32)
    delta = np.abs(actual_arr - reference_arr)
    denom = np.maximum(np.abs(reference_arr), 1e-8)
    return {
        "max_abs_error": float(np.max(delta)),
        "max_rel_error": float(np.max(delta / denom)),
    }
