from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np


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


@dataclass(slots=True)
class PageScoreWeights:
    similarity: float = 1.0
    prev_attention: float = 0.5
    distance: float = 0.3
    distance_decay: float = 0.001


@dataclass(slots=True)
class PagedAttentionControllerConfig:
    sink_window_tokens: int = 0
    recent_window_tokens: int = 0
    top_k: int = 0
    page_chunk_size: int = 4
    early_exit: bool = False
    early_exit_eps: float = 1e-4
    score_weights: PageScoreWeights = field(default_factory=PageScoreWeights)


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
class DecodeResult:
    output: np.ndarray
    processed_page_ids: list[int]
    pages_processed: int
    tokens_processed: int
    early_exit_triggered: bool
    early_exit_ratio: float | None
    m_final: float
    l_final: float
    logits: np.ndarray | None = None
    weights: np.ndarray | None = None


@dataclass(slots=True)
class PagedAttentionStepResult:
    output: np.ndarray
    selected_page_ids: list[int]
    processed_page_ids: list[int]
    score_time_ms: float
    selection_time_ms: float
    attention_time_ms: float
    total_step_time_ms: float
    selected_page_count: int
    processed_page_count: int
    tokens_processed: int
    early_exit_triggered: bool
    early_exit_ratio: float | None
    page_scores: np.ndarray


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


def _stack_selected_tokens_torch(
    resident: ResidentLayerPages,
    selected_page_ids: list[int],
):
    torch = _load_torch()
    k_chunks = []
    v_chunks = []
    for page_id in selected_page_ids:
        token_count = int(resident.page_token_counts_cpu[page_id])
        k_chunks.append(resident.k_pages[page_id, :token_count, :])
        v_chunks.append(resident.v_pages[page_id, :token_count, :])
    if not k_chunks:
        raise ValueError("selected_page_ids must be non-empty")
    return torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)


def decode_selected_pages_reference(
    query,
    *,
    k_pages,
    v_pages,
    selected_page_ids: list[int],
    page_token_counts,
) -> DecodeResult:
    if not selected_page_ids:
        raise ValueError("selected_page_ids must be non-empty")
    query_np = _to_numpy(query)
    k_pages_np = _to_numpy(k_pages)
    v_pages_np = _to_numpy(v_pages)
    page_token_counts_np = np.asarray(page_token_counts, dtype=np.int64)
    flat_k = np.concatenate(
        [k_pages_np[page_id, : int(page_token_counts_np[page_id]), :] for page_id in selected_page_ids],
        axis=0,
    ).astype(np.float32, copy=False)
    flat_v = np.concatenate(
        [v_pages_np[page_id, : int(page_token_counts_np[page_id]), :] for page_id in selected_page_ids],
        axis=0,
    ).astype(np.float32, copy=False)
    logits = flat_k @ query_np
    shifted = logits - np.max(logits)
    weights = np.exp(shifted).astype(np.float32, copy=False)
    weights = weights / np.sum(weights)
    output = weights @ flat_v
    return DecodeResult(
        output=output.astype(np.float32, copy=False),
        processed_page_ids=list(selected_page_ids),
        pages_processed=len(selected_page_ids),
        tokens_processed=int(logits.shape[0]),
        early_exit_triggered=False,
        early_exit_ratio=None,
        m_final=float(np.max(logits)),
        l_final=float(np.sum(np.exp(shifted.astype(np.float64)))),
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
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    with torch.no_grad():
        flat_k, flat_v = _stack_selected_tokens_torch(resident, selected_page_ids)
        logits = torch.matmul(flat_k.to(dtype=torch.float32), query_tensor)
        weights = torch.softmax(logits, dim=0)
        output = torch.matmul(weights.to(dtype=flat_v.dtype), flat_v).to(dtype=torch.float32)
    return DecodeResult(
        output=output.detach().cpu().numpy(),
        processed_page_ids=list(selected_page_ids),
        pages_processed=len(selected_page_ids),
        tokens_processed=int(logits.numel()),
        early_exit_triggered=False,
        early_exit_ratio=None,
        m_final=float(torch.max(logits).item()),
        l_final=float(torch.sum(torch.exp(logits - torch.max(logits))).item()),
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
    query_tensor = _to_device_tensor(query, device=resident.device, dtype=torch.float32)
    acc = resident.acc_buffer.zero_()
    logits_parts: list[np.ndarray] = []
    processed_page_ids: list[int] = []
    tokens_processed = 0
    early_exit_triggered = False
    early_exit_ratio: float | None = None
    m = None
    l = None
    with torch.no_grad():
        for page_start in range(0, len(selected_page_ids), page_chunk_size):
            chunk_page_ids = selected_page_ids[page_start : page_start + page_chunk_size]
            k_chunks = []
            v_chunks = []
            for page_id in chunk_page_ids:
                token_count = int(resident.page_token_counts_cpu[page_id])
                k_chunks.append(resident.k_pages[page_id, :token_count, :])
                v_chunks.append(resident.v_pages[page_id, :token_count, :])
            k_chunk = torch.cat(k_chunks, dim=0).to(dtype=torch.float32)
            v_chunk = torch.cat(v_chunks, dim=0)
            logits = torch.matmul(k_chunk, query_tensor)
            chunk_max = torch.max(logits)
            exp_scores = torch.exp(logits - chunk_max)
            chunk_sum = torch.sum(exp_scores)
            if m is None or l is None:
                m = chunk_max
                l = chunk_sum
                acc.copy_(torch.sum(exp_scores[:, None] * v_chunk.to(dtype=torch.float32), dim=0))
            else:
                m_new = torch.maximum(m, chunk_max)
                alpha = torch.exp(m - m_new)
                exp_scores = torch.exp(logits - m_new)
                chunk_sum = torch.sum(exp_scores)
                l = l * alpha + chunk_sum
                acc.mul_(alpha).add_(torch.sum(exp_scores[:, None] * v_chunk.to(dtype=torch.float32), dim=0))
                m = m_new
            tokens_processed += int(logits.numel())
            processed_page_ids.extend(chunk_page_ids)
            if return_debug:
                logits_parts.append(logits.detach().cpu().numpy())
            if early_exit and len(processed_page_ids) < len(selected_page_ids):
                ratio = float((chunk_sum / l).detach().cpu().item())
                if ratio <= float(early_exit_eps):
                    early_exit_triggered = True
                    early_exit_ratio = ratio
                    break

    if m is None or l is None:
        raise RuntimeError("failed to process any attention chunks")
    output = (acc / l).detach().cpu().numpy().astype(np.float32, copy=False)
    logits_np = None
    weights_np = None
    if return_debug and logits_parts:
        logits_np = np.concatenate(logits_parts, axis=0).astype(np.float32, copy=False)
        shifted = logits_np - np.max(logits_np)
        weights_np = np.exp(shifted).astype(np.float32, copy=False)
        weights_np = (weights_np / np.sum(weights_np)).astype(np.float32, copy=False)
    return DecodeResult(
        output=output,
        processed_page_ids=processed_page_ids,
        pages_processed=len(processed_page_ids),
        tokens_processed=tokens_processed,
        early_exit_triggered=early_exit_triggered,
        early_exit_ratio=early_exit_ratio,
        m_final=float(m.detach().cpu().item()),
        l_final=float(l.detach().cpu().item()),
        logits=logits_np,
        weights=weights_np,
    )


def run_paged_attention_step(
    query,
    resident: ResidentLayerPages,
    *,
    config: PagedAttentionControllerConfig,
    engine: Literal["mps_experimental", "torch_mps_baseline"] = "mps_experimental",
) -> PagedAttentionStepResult:
    total_start = time.perf_counter()

    score_start = time.perf_counter()
    page_scores = score_pages_mps(query, resident, score_weights=config.score_weights)
    _sync_device(resident.device)
    score_time_ms = (time.perf_counter() - score_start) * 1000.0

    selection_start = time.perf_counter()
    selection = _select_page_indices_from_scores(
        page_scores.detach().cpu().numpy(),
        page_token_starts=resident.page_token_starts_cpu,
        page_token_counts=resident.page_token_counts_cpu,
        config=config,
    )
    selection_time_ms = (time.perf_counter() - selection_start) * 1000.0

    attention_start = time.perf_counter()
    if engine == "mps_experimental":
        decode_result = decode_selected_pages_mps(
            query,
            resident,
            selection.selected_page_ids,
            page_chunk_size=config.page_chunk_size,
            early_exit=config.early_exit,
            early_exit_eps=config.early_exit_eps,
        )
    elif engine == "torch_mps_baseline":
        decode_result = decode_selected_pages_dense_mps(query, resident, selection.selected_page_ids)
    else:
        raise ValueError(f"unsupported engine: {engine}")
    _sync_device(resident.device)
    attention_time_ms = (time.perf_counter() - attention_start) * 1000.0

    total_step_time_ms = (time.perf_counter() - total_start) * 1000.0
    return PagedAttentionStepResult(
        output=decode_result.output,
        selected_page_ids=selection.selected_page_ids,
        processed_page_ids=decode_result.processed_page_ids,
        score_time_ms=score_time_ms,
        selection_time_ms=selection_time_ms,
        attention_time_ms=attention_time_ms,
        total_step_time_ms=total_step_time_ms,
        selected_page_count=len(selection.selected_page_ids),
        processed_page_count=decode_result.pages_processed,
        tokens_processed=decode_result.tokens_processed,
        early_exit_triggered=decode_result.early_exit_triggered,
        early_exit_ratio=decode_result.early_exit_ratio,
        page_scores=selection.page_scores,
    )


def run_reference_step(
    snapshot: PagedAttentionSnapshot,
    *,
    config: PagedAttentionControllerConfig,
) -> PagedAttentionStepResult:
    total_start = time.perf_counter()

    score_start = time.perf_counter()
    page_scores = score_pages_reference(
        snapshot.query,
        snapshot.page_k_mean,
        snapshot.prev_attn,
        snapshot.distance,
        score_weights=config.score_weights,
    )
    score_time_ms = (time.perf_counter() - score_start) * 1000.0

    selection_start = time.perf_counter()
    selection = _select_page_indices_from_scores(
        page_scores,
        page_token_starts=snapshot.page_token_starts,
        page_token_counts=snapshot.page_token_counts,
        config=config,
    )
    selection_time_ms = (time.perf_counter() - selection_start) * 1000.0

    attention_start = time.perf_counter()
    decode_result = decode_selected_pages_reference(
        snapshot.query,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        selected_page_ids=selection.selected_page_ids,
        page_token_counts=snapshot.page_token_counts,
    )
    attention_time_ms = (time.perf_counter() - attention_start) * 1000.0

    total_step_time_ms = (time.perf_counter() - total_start) * 1000.0
    return PagedAttentionStepResult(
        output=decode_result.output,
        selected_page_ids=selection.selected_page_ids,
        processed_page_ids=decode_result.processed_page_ids,
        score_time_ms=score_time_ms,
        selection_time_ms=selection_time_ms,
        attention_time_ms=attention_time_ms,
        total_step_time_ms=total_step_time_ms,
        selected_page_count=len(selection.selected_page_ids),
        processed_page_count=decode_result.pages_processed,
        tokens_processed=decode_result.tokens_processed,
        early_exit_triggered=False,
        early_exit_ratio=None,
        page_scores=selection.page_scores,
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
