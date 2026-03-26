from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..tracing import ExecutionTrace
from ..types import EncodedPage, PageHeader

_UNPACK_METADATA: dict[int, tuple[Any, Any]] = {}
_MAX_PREPARE_PAGES_PER_CHUNK = 128


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


@dataclass(slots=True)
class PreparedPageMPS:
    source_page: EncodedPage
    header: PageHeader
    payload: Any | None = None
    scales: Any | None = None
    bias: Any | None = None
    escape_payload: Any | None = None
    unpack_shifts: Any | None = None
    unpack_mask: Any | None = None
    host_to_device_nbytes: int = 0
    resident_nbytes: int = 0

    @property
    def payload_nbytes(self) -> int:
        return self.source_page.payload_nbytes

    @property
    def metadata_nbytes(self) -> int:
        return self.source_page.metadata_nbytes


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in environments without torch
        raise RuntimeError("torch is required for the torch_mps backend") from exc
    return torch


def _device_tensor(array: np.ndarray, *, device: str):
    torch = _load_torch()
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


def _prepare_signature(page: EncodedPage | PreparedPageMPS) -> tuple[int | str, ...]:
    source_page = page.source_page if isinstance(page, PreparedPageMPS) else page
    header = source_page.header
    return (
        header.kind,
        header.mode_default,
        header.token_count,
        header.head_dim,
        header.padded_head_dim,
        header.group_size,
        header.num_groups,
        header.bits,
        header.words_per_group,
        header.layout,
        header.quant_scheme,
        header.escape_dtype,
    )


def _chunk_compatible_source_pages(
    pages: Sequence[EncodedPage | PreparedPageMPS],
) -> list[list[EncodedPage | PreparedPageMPS]]:
    chunks: list[list[EncodedPage | PreparedPageMPS]] = []
    current_chunk: list[EncodedPage | PreparedPageMPS] = []
    current_signature: tuple[int | str, ...] | None = None
    for page in pages:
        signature = _prepare_signature(page)
        if current_chunk and (
            signature != current_signature or len(current_chunk) >= _MAX_PREPARE_PAGES_PER_CHUNK
        ):
            chunks.append(current_chunk)
            current_chunk = [page]
            current_signature = signature
            continue
        if not current_chunk:
            current_signature = signature
        current_chunk.append(page)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def page_supported_mps(page: EncodedPage | PreparedPageMPS) -> bool:
    source_page = page.source_page if isinstance(page, PreparedPageMPS) else page
    header = source_page.header
    if header.layout != "group_major":
        return False
    if header.mode_default == "M3":
        return source_page.escape_payload is not None
    return (
        header.mode_default == "M0"
        and header.bits in (2, 4)
        and header.group_size in (32, 64)
        and header.quant_scheme == "affine"
        and source_page.payload is not None
        and source_page.scales is not None
        and source_page.bias is not None
    )


def _unpack_metadata(bits: int):
    cached = _UNPACK_METADATA.get(bits)
    if cached is not None:
        return cached
    torch = _load_torch()
    symbols_per_word = 32 // bits
    shifts = torch.arange(symbols_per_word, dtype=torch.int32, device="mps") * bits
    mask = torch.tensor((1 << bits) - 1, dtype=torch.int32, device="mps")
    _UNPACK_METADATA[bits] = (shifts, mask)
    return shifts, mask


def _prepared_page_host_nbytes(page: EncodedPage) -> int:
    total = 0
    if page.payload is not None:
        total += int(page.payload.nbytes)
    if page.scales is not None:
        total += int(page.scales.nbytes)
    if page.bias is not None:
        total += int(page.bias.nbytes)
    if page.escape_payload is not None:
        total += int(page.escape_payload.nbytes)
    return total


def _prepare_page_chunk_mps(
    pages: Sequence[EncodedPage],
    *,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageMPS]:
    torch = _load_torch()
    if not pages:
        return []
    header = pages[0].header
    total_host_to_device_nbytes = 0

    if header.mode_default == "M3":
        escape_batch = _device_tensor(np.stack([np.asarray(page.escape_payload) for page in pages], axis=0), device="mps")
        total_host_to_device_nbytes += int(escape_batch.numel() * escape_batch.element_size())
        prepared_pages = [
            PreparedPageMPS(
                source_page=page,
                header=page.header,
                escape_payload=escape_batch[index],
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=int(escape_batch[index].numel() * escape_batch[index].element_size()),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    payload_batch = _device_tensor(np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0), device="mps")
    scales_batch = _device_tensor(np.stack([np.asarray(page.scales) for page in pages], axis=0), device="mps")
    bias_batch = _device_tensor(np.stack([np.asarray(page.bias) for page in pages], axis=0), device="mps")
    scales_batch_fp32 = scales_batch.to(dtype=torch.float32)
    bias_batch_fp32 = bias_batch.to(dtype=torch.float32)
    total_host_to_device_nbytes += int(payload_batch.numel() * payload_batch.element_size())
    total_host_to_device_nbytes += int(scales_batch.numel() * scales_batch.element_size())
    total_host_to_device_nbytes += int(bias_batch.numel() * bias_batch.element_size())
    unpack_shifts, unpack_mask = _unpack_metadata(header.bits)

    prepared_pages = [
        PreparedPageMPS(
            source_page=page,
            header=page.header,
            payload=payload_batch[index],
            scales=scales_batch_fp32[index],
            bias=bias_batch_fp32[index],
            unpack_shifts=unpack_shifts,
            unpack_mask=unpack_mask,
            host_to_device_nbytes=_prepared_page_host_nbytes(page),
            resident_nbytes=(
                int(payload_batch[index].numel() * payload_batch[index].element_size())
                + int(scales_batch_fp32[index].numel() * scales_batch_fp32[index].element_size())
                + int(bias_batch_fp32[index].numel() * bias_batch_fp32[index].element_size())
            ),
        )
        for index, page in enumerate(pages)
    ]
    if trace is not None:
        trace.record_host_to_device(total_host_to_device_nbytes)
    return prepared_pages


def prepare_pages_mps(
    pages: Sequence[EncodedPage | PreparedPageMPS],
    *,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageMPS]:
    if not mps_available():
        raise RuntimeError("torch_mps is unavailable on this machine")
    prepared_pages: list[PreparedPageMPS] = []
    for page_chunk in _chunk_compatible_source_pages(pages):
        if all(isinstance(page, PreparedPageMPS) for page in page_chunk):
            prepared_pages.extend(page_chunk)
            continue
        if any(isinstance(page, PreparedPageMPS) for page in page_chunk):
            prepared_pages.extend(prepare_page_mps(page, trace=trace) for page in page_chunk)
            continue
        source_pages = []
        for page in page_chunk:
            if not page_supported_mps(page):
                raise ValueError("page is unsupported by torch_mps in this phase")
            source_pages.append(page)
        if source_pages:
            prepared_pages.extend(_prepare_page_chunk_mps(source_pages, trace=trace))
    return prepared_pages


def prepare_page_mps(page: EncodedPage | PreparedPageMPS, *, trace: ExecutionTrace | None = None) -> PreparedPageMPS:
    if isinstance(page, PreparedPageMPS):
        return page
    return prepare_pages_mps([page], trace=trace)[0]


def _pad_query(query_slice: np.ndarray, padded_head_dim: int):
    torch = _load_torch()
    query = torch.as_tensor(query_slice, dtype=torch.float32, device="mps")
    if query.ndim != 1:
        raise ValueError("query_slice must have shape [head_dim]")
    if int(query.shape[0]) > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if int(query.shape[0]) == padded_head_dim:
        return query
    padded = torch.zeros(padded_head_dim, dtype=torch.float32, device="mps")
    padded[: query.shape[0]] = query
    return padded


def _pad_queries(query_slices: np.ndarray, padded_head_dim: int):
    torch = _load_torch()
    queries = torch.as_tensor(query_slices, dtype=torch.float32, device="mps")
    if queries.ndim != 2:
        raise ValueError("query_slices must have shape [query_count, head_dim]")
    if int(queries.shape[1]) > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if int(queries.shape[1]) == padded_head_dim:
        return queries
    padded = torch.zeros((queries.shape[0], padded_head_dim), dtype=torch.float32, device="mps")
    padded[:, : queries.shape[1]] = queries
    return padded


def _prepare_output_accumulator(out_acc: np.ndarray | None, head_dim: int, padded_head_dim: int):
    torch = _load_torch()
    output = torch.zeros(padded_head_dim, dtype=torch.float32, device="mps")
    if out_acc is None:
        return output
    values = torch.as_tensor(out_acc, dtype=torch.float32, device="mps")
    if values.shape != (head_dim,):
        raise ValueError("out_acc must have shape [head_dim]")
    output[:head_dim] = values
    return output


def _prepare_output_accumulator_tensor(out_acc, head_dim: int, padded_head_dim: int):
    torch = _load_torch()
    if out_acc is None:
        return torch.zeros(padded_head_dim, dtype=torch.float32, device="mps")
    if isinstance(out_acc, np.ndarray):
        return _prepare_output_accumulator(out_acc, head_dim, padded_head_dim)
    if tuple(out_acc.shape) != (padded_head_dim,):
        raise ValueError("out_acc tensor must have shape [padded_head_dim]")
    return out_acc


def _unpack_bits_torch(words, shifts, mask, group_size: int):
    torch = _load_torch()
    if words.ndim != 2:
        raise ValueError("words must have shape [token_count, words_per_group]")
    if shifts is None or mask is None:
        raise ValueError("prepared MPS pages require unpack metadata")
    expanded = torch.bitwise_and(torch.bitwise_right_shift(words[..., None], shifts), mask)
    return expanded.reshape(words.shape[0], group_size).to(torch.float32)


def _batched_signature(page: PreparedPageMPS) -> tuple[int | str, ...]:
    header = page.header
    return (
        header.kind,
        header.mode_default,
        header.token_count,
        header.head_dim,
        header.padded_head_dim,
        header.group_size,
        header.num_groups,
        header.bits,
        header.words_per_group,
        header.layout,
        header.quant_scheme,
    )


def _chunk_compatible_pages(pages: Sequence[PreparedPageMPS]) -> list[list[PreparedPageMPS]]:
    chunks: list[list[PreparedPageMPS]] = []
    current_chunk: list[PreparedPageMPS] = []
    current_signature: tuple[int | str, ...] | None = None
    for page in pages:
        signature = _batched_signature(page)
        if current_chunk and signature != current_signature:
            chunks.append(current_chunk)
            current_chunk = [page]
            current_signature = signature
            continue
        if not current_chunk:
            current_signature = signature
        current_chunk.append(page)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _score_page_chunk_mps(query_slice: np.ndarray, pages: Sequence[PreparedPageMPS], *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if header.mode_default == "M3":
        dense = torch.stack(
            [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in pages],
            dim=0,
        )
        query = torch.as_tensor(query_slice, dtype=torch.float32, device="mps")
        return torch.matmul(dense, query).reshape(-1)

    query = _pad_query(query_slice, header.padded_head_dim)
    query_groups = query.reshape(header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    page_count = len(pages)
    logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device="mps")

    for group_index in range(header.num_groups):
        group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
        codes = _unpack_bits_torch(
            group_words.reshape(-1, header.words_per_group),
            pages[0].unpack_shifts,
            pages[0].unpack_mask,
            header.group_size,
        ).reshape(page_count, header.token_count, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[group_index]
        int_dot = torch.matmul(codes, qg)
        scales = torch.stack([page.scales[:, group_index] for page in pages], dim=0)
        bias = torch.stack([page.bias[:, group_index] for page in pages], dim=0)
        logits += scales * int_dot + bias * query_group_sums[group_index]

    return logits.reshape(-1)


def _mix_page_chunk_mps(
    attn_weights,
    pages: Sequence[PreparedPageMPS],
    *,
    out_acc=None,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    page_count = len(pages)
    token_count = header.token_count
    output = _prepare_output_accumulator_tensor(out_acc, header.head_dim, header.padded_head_dim)

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if not isinstance(attn_weights, torch.Tensor):
        weights = torch.as_tensor(attn_weights, dtype=torch.float32, device="mps")
    else:
        weights = attn_weights
    expected_shape = (page_count, token_count)
    if tuple(weights.shape) != expected_shape:
        raise ValueError("attn_weights chunk must have shape [page_count, token_count]")

    if header.mode_default == "M3":
        dense = torch.stack(
            [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in pages],
            dim=0,
        )
        output[: header.head_dim] += torch.sum(weights[..., None] * dense, dim=(0, 1))
        return output

    for group_index in range(header.num_groups):
        group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
        codes = _unpack_bits_torch(
            group_words.reshape(-1, header.words_per_group),
            pages[0].unpack_shifts,
            pages[0].unpack_mask,
            header.group_size,
        ).reshape(page_count, token_count, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = torch.stack([page.scales[:, group_index] for page in pages], dim=0)
        bias = torch.stack([page.bias[:, group_index] for page in pages], dim=0)
        weighted_scales = weights * scales
        contribution = torch.sum(weighted_scales[..., None] * codes, dim=(0, 1))
        bias_term = torch.sum(weights * bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[start:end] += contribution + bias_term

    return output


def score_page_mps(
    query_slice: np.ndarray,
    page: EncodedPage | PreparedPageMPS,
    *,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    torch = _load_torch()
    prepared = prepare_page_mps(page, trace=trace)
    header = prepared.header

    if trace is not None:
        trace.record_page_read(prepared.payload_nbytes, prepared.metadata_nbytes)

    if header.mode_default == "M3":
        if prepared.escape_payload is None:
            raise ValueError("escape payload is missing")
        dense = prepared.escape_payload[: header.token_count, : header.head_dim].to(torch.float32)
        query = torch.as_tensor(query_slice, dtype=torch.float32, device="mps")
        return torch.matmul(dense, query).detach().cpu().numpy()

    if prepared.payload is None or prepared.scales is None or prepared.bias is None:
        raise ValueError("M0 page is missing payload or affine metadata")

    query = _pad_query(query_slice, header.padded_head_dim)
    query_groups = query.reshape(header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    logits = torch.zeros(header.token_count, dtype=torch.float32, device="mps")

    for group_index in range(header.num_groups):
        group_words = prepared.payload[group_index]
        codes = _unpack_bits_torch(group_words, prepared.unpack_shifts, prepared.unpack_mask, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[group_index]
        int_dot = torch.matmul(codes, qg)
        scales = prepared.scales[:, group_index]
        bias = prepared.bias[:, group_index]
        logits += scales * int_dot + bias * query_group_sums[group_index]

    return logits.detach().cpu().numpy()


def score_pages_mps(
    query_slice: np.ndarray,
    pages: Sequence[EncodedPage | PreparedPageMPS],
    *,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    prepared_pages = [prepare_page_mps(page, trace=trace) for page in pages]
    if not prepared_pages:
        return []

    page_logits: list[np.ndarray] = []
    for page_chunk in _chunk_compatible_pages(prepared_pages):
        chunk_logits = _score_page_chunk_mps(query_slice, page_chunk, trace=trace)
        chunk_logits = chunk_logits.reshape(len(page_chunk), page_chunk[0].header.token_count)
        page_logits.extend(
            chunk_logits[index].detach().cpu().numpy()
            for index in range(len(page_chunk))
        )
    return page_logits


def mix_page_mps(
    attn_weights: np.ndarray,
    page: EncodedPage | PreparedPageMPS,
    *,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    torch = _load_torch()
    prepared = prepare_page_mps(page, trace=trace)
    header = prepared.header

    if trace is not None:
        trace.record_page_read(prepared.payload_nbytes, prepared.metadata_nbytes)

    weights = torch.as_tensor(attn_weights, dtype=torch.float32, device="mps")
    if weights.shape != (header.token_count,):
        raise ValueError("attn_weights must have shape [token_count]")

    if header.mode_default == "M3":
        if prepared.escape_payload is None:
            raise ValueError("escape payload is missing")
        dense = prepared.escape_payload[: header.token_count, : header.head_dim].to(torch.float32)
        output = torch.matmul(weights, dense)
        if out_acc is not None:
            base = torch.as_tensor(out_acc, dtype=torch.float32, device="mps")
            if base.shape != (header.head_dim,):
                raise ValueError("out_acc must have shape [head_dim]")
            output = output + base
        return output.detach().cpu().numpy()

    if prepared.payload is None or prepared.scales is None or prepared.bias is None:
        raise ValueError("M0 page is missing payload or affine metadata")

    output = _prepare_output_accumulator(out_acc, header.head_dim, header.padded_head_dim)

    for group_index in range(header.num_groups):
        group_words = prepared.payload[group_index]
        codes = _unpack_bits_torch(group_words, prepared.unpack_shifts, prepared.unpack_mask, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        weighted_scales = weights * prepared.scales[:, group_index]
        bias_term = torch.sum(weights * prepared.bias[:, group_index])
        contribution = torch.matmul(weighted_scales, codes)
        start = group_index * header.group_size
        end = start + header.group_size
        output[start:end] += contribution + bias_term

    return output[: header.head_dim].detach().cpu().numpy()


def _page_logits_tensor(page_logits, token_count: int):
    torch = _load_torch()
    logits = torch.as_tensor(page_logits, dtype=torch.float32, device="mps")
    if tuple(logits.shape) != (token_count,):
        raise ValueError("precomputed page logits must have shape [token_count]")
    return logits


def decode_step_mps(
    query_slice: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageMPS],
    value_pages: Sequence[EncodedPage | PreparedPageMPS],
    *,
    precomputed_page_logits: Sequence[np.ndarray | Any | None] | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = _load_torch()
    prepared_key_pages = [prepare_page_mps(page, trace=trace) for page in key_pages]
    prepared_value_pages = [prepare_page_mps(page, trace=trace) for page in value_pages]
    if not prepared_key_pages:
        raise ValueError("decode_step_mps requires at least one page")
    if precomputed_page_logits is not None and len(precomputed_page_logits) != len(prepared_key_pages):
        raise ValueError("precomputed_page_logits must align with key_pages")

    logits_parts = []
    score_run: list[PreparedPageMPS] = []

    def flush_score_run() -> None:
        nonlocal score_run
        if not score_run:
            return
        chunk_logits = _score_page_chunk_mps(query_slice, score_run, trace=trace)
        chunk_logits = chunk_logits.reshape(len(score_run), score_run[0].header.token_count)
        logits_parts.extend(chunk_logits[index] for index in range(len(score_run)))
        score_run = []

    for index, page in enumerate(prepared_key_pages):
        cached_logits = None if precomputed_page_logits is None else precomputed_page_logits[index]
        if cached_logits is not None:
            flush_score_run()
            logits_parts.append(_page_logits_tensor(cached_logits, page.header.token_count))
            continue
        if score_run and _batched_signature(score_run[-1]) != _batched_signature(page):
            flush_score_run()
        score_run.append(page)
    flush_score_run()

    logits = torch.cat(logits_parts, dim=0)
    weights = torch.softmax(logits, dim=0)

    output = torch.zeros(
        prepared_value_pages[0].header.padded_head_dim,
        dtype=torch.float32,
        device="mps",
    )
    offset = 0
    for page_chunk in _chunk_compatible_pages(prepared_value_pages):
        chunk_token_count = page_chunk[0].header.token_count * len(page_chunk)
        chunk_weights = weights[offset : offset + chunk_token_count].reshape(len(page_chunk), page_chunk[0].header.token_count)
        output = _mix_page_chunk_mps(chunk_weights, page_chunk, out_acc=output, trace=trace)
        offset += chunk_token_count

    head_dim = prepared_value_pages[0].header.head_dim
    return (
        logits.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        output[:head_dim].detach().cpu().numpy(),
    )


def _score_page_chunk_multiquery_mps(
    query_slices: np.ndarray,
    pages: Sequence[PreparedPageMPS],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    if torch.is_tensor(query_slices):
        query_count = int(query_slices.shape[0])
    else:
        query_count = int(np.asarray(query_slices).shape[0])
    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if header.mode_default == "M3":
        dense = torch.stack(
            [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in pages],
            dim=0,
        )
        queries = torch.as_tensor(query_slices, dtype=torch.float32, device="mps")
        return torch.einsum("pth,qh->qpt", dense, queries).reshape(query_count, -1)

    queries = _pad_queries(query_slices, header.padded_head_dim)
    query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    page_count = len(pages)
    logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device="mps")

    for group_index in range(header.num_groups):
        group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
        codes = _unpack_bits_torch(
            group_words.reshape(-1, header.words_per_group),
            pages[0].unpack_shifts,
            pages[0].unpack_mask,
            header.group_size,
        ).reshape(page_count, header.token_count, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[:, group_index, :]
        int_dot = torch.einsum("ptg,qg->qpt", codes, qg)
        scales = torch.stack([page.scales[:, group_index] for page in pages], dim=0)
        bias = torch.stack([page.bias[:, group_index] for page in pages], dim=0)
        logits += scales.unsqueeze(0) * int_dot + bias.unsqueeze(0) * query_group_sums[:, group_index].reshape(
            query_count,
            1,
            1,
        )

    return logits.reshape(query_count, -1)


def _mix_page_chunk_multiquery_mps(
    attn_weights,
    pages: Sequence[PreparedPageMPS],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    page_count = len(pages)
    token_count = header.token_count

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if not isinstance(attn_weights, torch.Tensor):
        weights = torch.as_tensor(attn_weights, dtype=torch.float32, device="mps")
    else:
        weights = attn_weights
    if weights.ndim != 3 or tuple(weights.shape[1:]) != (page_count, token_count):
        raise ValueError("attn_weights chunk must have shape [query_count, page_count, token_count]")

    query_count = int(weights.shape[0])
    output = torch.zeros((query_count, header.padded_head_dim), dtype=torch.float32, device="mps")

    if header.mode_default == "M3":
        dense = torch.stack(
            [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in pages],
            dim=0,
        )
        output[:, : header.head_dim] += torch.einsum("qpt,pth->qh", weights, dense)
        return output

    for group_index in range(header.num_groups):
        group_words = torch.stack([page.payload[group_index] for page in pages], dim=0)
        codes = _unpack_bits_torch(
            group_words.reshape(-1, header.words_per_group),
            pages[0].unpack_shifts,
            pages[0].unpack_mask,
            header.group_size,
        ).reshape(page_count, token_count, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = torch.stack([page.scales[:, group_index] for page in pages], dim=0)
        bias = torch.stack([page.bias[:, group_index] for page in pages], dim=0)
        weighted_scales = weights * scales.unsqueeze(0)
        contribution = torch.einsum("qpt,ptg->qg", weighted_scales, codes)
        bias_term = torch.einsum("qpt,pt->q", weights, bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[:, start:end] += contribution + bias_term[:, None]

    return output


def _score_page_chunk_grouped_multiquery_mps(
    query_groups,
    pages_by_group: Sequence[Sequence[PreparedPageMPS]],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages_by_group or not pages_by_group[0]:
        raise ValueError("pages_by_group must be non-empty")
    batch_size = len(pages_by_group)
    page_count = len(pages_by_group[0])
    header = pages_by_group[0][0].header

    for group_pages in pages_by_group:
        if len(group_pages) != page_count:
            raise ValueError("all page groups must have the same page count")
        for page in group_pages:
            if _batched_signature(page) != _batched_signature(pages_by_group[0][0]):
                raise ValueError("all grouped pages must share the same page signature within a chunk")

    queries = torch.stack(
        [
            group.to(dtype=torch.float32, device="mps") if torch.is_tensor(group) else torch.as_tensor(group, dtype=torch.float32, device="mps")
            for group in query_groups
        ],
        dim=0,
    )
    query_count = int(queries.shape[1])

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for group_pages in pages_by_group for page in group_pages),
            sum(page.metadata_nbytes for group_pages in pages_by_group for page in group_pages),
        )

    if header.mode_default == "M3":
        dense = torch.stack(
            [
                torch.stack(
                    [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in group_pages],
                    dim=0,
                )
                for group_pages in pages_by_group
            ],
            dim=0,
        )
        return torch.einsum("bpth,bqh->bqpt", dense, queries).reshape(batch_size, query_count, -1)

    padded_queries = _pad_queries(queries.reshape(batch_size * query_count, header.head_dim), header.padded_head_dim).reshape(
        batch_size,
        query_count,
        header.padded_head_dim,
    )
    query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
    query_group_sums = query_groups_tensor.sum(dim=-1)
    logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device="mps")

    for group_index in range(header.num_groups):
        group_words = torch.stack(
            [torch.stack([page.payload[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
            dim=0,
        )
        codes = _unpack_bits_torch(
            group_words.reshape(-1, header.words_per_group),
            pages_by_group[0][0].unpack_shifts,
            pages_by_group[0][0].unpack_mask,
            header.group_size,
        ).reshape(batch_size, page_count, header.token_count, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups_tensor[:, :, group_index, :]
        int_dot = torch.einsum("bptg,bqg->bqpt", codes, qg)
        scales = torch.stack(
            [torch.stack([page.scales[:, group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
            dim=0,
        )
        bias = torch.stack(
            [torch.stack([page.bias[:, group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
            dim=0,
        )
        logits += scales[:, None] * int_dot + bias[:, None] * query_group_sums[:, :, group_index].reshape(
            batch_size,
            query_count,
            1,
            1,
        )

    return logits.reshape(batch_size, query_count, -1)


def _mix_page_chunk_grouped_multiquery_mps(
    attn_weights,
    pages_by_group: Sequence[Sequence[PreparedPageMPS]],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages_by_group or not pages_by_group[0]:
        raise ValueError("pages_by_group must be non-empty")
    batch_size = len(pages_by_group)
    page_count = len(pages_by_group[0])
    header = pages_by_group[0][0].header
    token_count = header.token_count

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for group_pages in pages_by_group for page in group_pages),
            sum(page.metadata_nbytes for group_pages in pages_by_group for page in group_pages),
        )

    weights = attn_weights if isinstance(attn_weights, torch.Tensor) else torch.as_tensor(attn_weights, dtype=torch.float32, device="mps")
    if weights.ndim != 4 or tuple(weights.shape[2:]) != (page_count, token_count):
        raise ValueError("grouped attn_weights chunk must have shape [batch_size, query_count, page_count, token_count]")

    query_count = int(weights.shape[1])
    output = torch.zeros((batch_size, query_count, header.padded_head_dim), dtype=torch.float32, device="mps")

    if header.mode_default == "M3":
        dense = torch.stack(
            [
                torch.stack(
                    [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in group_pages],
                    dim=0,
                )
                for group_pages in pages_by_group
            ],
            dim=0,
        )
        output[:, :, : header.head_dim] += torch.einsum("bqpt,bpth->bqh", weights, dense)
        return output

    for group_index in range(header.num_groups):
        group_words = torch.stack(
            [torch.stack([page.payload[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
            dim=0,
        )
        codes = _unpack_bits_torch(
            group_words.reshape(-1, header.words_per_group),
            pages_by_group[0][0].unpack_shifts,
            pages_by_group[0][0].unpack_mask,
            header.group_size,
        ).reshape(batch_size, page_count, token_count, header.group_size)
        if trace is not None:
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = torch.stack(
            [torch.stack([page.scales[:, group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
            dim=0,
        )
        bias = torch.stack(
            [torch.stack([page.bias[:, group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
            dim=0,
        )
        weighted_scales = weights * scales[:, None]
        contribution = torch.einsum("bqpt,bptg->bqg", weighted_scales, codes)
        bias_term = torch.einsum("bqpt,bpt->bq", weights, bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[:, :, start:end] += contribution + bias_term[:, :, None]

    return output


def decode_multi_query_step_mps_tensor(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageMPS],
    value_pages: Sequence[EncodedPage | PreparedPageMPS],
    *,
    trace: ExecutionTrace | None = None,
) -> tuple:
    torch = _load_torch()
    if all(isinstance(page, PreparedPageMPS) for page in key_pages):
        prepared_key_pages = list(key_pages)
    else:
        prepared_key_pages = [prepare_page_mps(page, trace=trace) for page in key_pages]
    if all(isinstance(page, PreparedPageMPS) for page in value_pages):
        prepared_value_pages = list(value_pages)
    else:
        prepared_value_pages = [prepare_page_mps(page, trace=trace) for page in value_pages]
    if not prepared_key_pages:
        raise ValueError("decode_multi_query_step_mps requires at least one page")

    logits_parts = []
    for page_chunk in _chunk_compatible_pages(prepared_key_pages):
        logits_parts.append(_score_page_chunk_multiquery_mps(query_slices, page_chunk, trace=trace))
    logits = torch.cat(logits_parts, dim=1)
    weights = torch.softmax(logits, dim=1)

    output = torch.zeros(
        (
            int(query_slices.shape[0]) if torch.is_tensor(query_slices) else int(np.asarray(query_slices).shape[0]),
            prepared_value_pages[0].header.padded_head_dim,
        ),
        dtype=torch.float32,
        device="mps",
    )
    offset = 0
    for page_chunk in _chunk_compatible_pages(prepared_value_pages):
        chunk_token_count = page_chunk[0].header.token_count * len(page_chunk)
        chunk_weights = weights[:, offset : offset + chunk_token_count].reshape(
            weights.shape[0],
            len(page_chunk),
            page_chunk[0].header.token_count,
        )
        output += _mix_page_chunk_multiquery_mps(chunk_weights, page_chunk, trace=trace)
        offset += chunk_token_count

    head_dim = prepared_value_pages[0].header.head_dim
    return logits, weights, output[:, :head_dim]


def decode_grouped_multiquery_step_mps_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageMPS]],
    value_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageMPS]],
    *,
    trace: ExecutionTrace | None = None,
    ):
    torch = _load_torch()
    if not key_pages_by_group or not value_pages_by_group:
        raise ValueError("grouped decode requires non-empty key/value page groups")
    if len(key_pages_by_group) != len(value_pages_by_group):
        raise ValueError("key_pages_by_group and value_pages_by_group must have the same group count")
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        raise ValueError("query_groups must align with key/value group count")

    prepared_key_groups = [
        list(group_pages) if all(isinstance(page, PreparedPageMPS) for page in group_pages) else [prepare_page_mps(page, trace=trace) for page in group_pages]
        for group_pages in key_pages_by_group
    ]
    prepared_value_groups = [
        list(group_pages) if all(isinstance(page, PreparedPageMPS) for page in group_pages) else [prepare_page_mps(page, trace=trace) for page in group_pages]
        for group_pages in value_pages_by_group
    ]
    if not prepared_key_groups[0]:
        raise ValueError("grouped decode requires at least one key page per group")

    query_tensors = [
        group.to(dtype=torch.float32, device="mps") if torch.is_tensor(group) else torch.as_tensor(group, dtype=torch.float32, device="mps")
        for group in query_groups
    ]
    query_count = int(query_tensors[0].shape[0])
    for group_query in query_tensors:
        if int(group_query.shape[0]) != query_count:
            raise ValueError("all query groups must have the same query count for batched grouped decode")

    page_count = len(prepared_key_groups[0])
    for group_pages in prepared_key_groups[1:]:
        if len(group_pages) != page_count:
            raise ValueError("all key page groups must have the same page count for batched grouped decode")
    for group_pages in prepared_value_groups[1:]:
        if len(group_pages) != page_count:
            raise ValueError("all value page groups must have the same page count for batched grouped decode")

    for page_index in range(page_count):
        page_signature = _batched_signature(prepared_key_groups[0][page_index])
        value_signature = _batched_signature(prepared_value_groups[0][page_index])
        for group_index in range(1, group_count):
            if _batched_signature(prepared_key_groups[group_index][page_index]) != page_signature:
                raise ValueError("grouped key pages must share aligned signatures for batched grouped decode")
            if _batched_signature(prepared_value_groups[group_index][page_index]) != value_signature:
                raise ValueError("grouped value pages must share aligned signatures for batched grouped decode")

    key_chunks = [_chunk_compatible_pages(group_pages) for group_pages in prepared_key_groups]
    value_chunks = [_chunk_compatible_pages(group_pages) for group_pages in prepared_value_groups]
    chunk_count = len(key_chunks[0])
    if any(len(chunks) != chunk_count for chunks in key_chunks[1:]):
        raise ValueError("grouped key page chunks must align for batched grouped decode")
    if any(len(chunks) != len(value_chunks[0]) for chunks in value_chunks[1:]):
        raise ValueError("grouped value page chunks must align for batched grouped decode")

    logits_parts = []
    for chunk_index in range(chunk_count):
        chunk_pages = [chunks[chunk_index] for chunks in key_chunks]
        logits_parts.append(_score_page_chunk_grouped_multiquery_mps(query_tensors, chunk_pages, trace=trace))
    logits = torch.cat(logits_parts, dim=2)
    weights = torch.softmax(logits, dim=2)

    head_dim = prepared_value_groups[0][0].header.head_dim
    padded_head_dim = prepared_value_groups[0][0].header.padded_head_dim
    output = torch.zeros((group_count, query_count, padded_head_dim), dtype=torch.float32, device="mps")
    offset = 0
    for chunk_index, chunk_template in enumerate(value_chunks[0]):
        page_chunk_count = len(chunk_template)
        chunk_token_count = chunk_template[0].header.token_count * page_chunk_count
        chunk_weights = weights[:, :, offset : offset + chunk_token_count].reshape(
            group_count,
            query_count,
            page_chunk_count,
            chunk_template[0].header.token_count,
        )
        chunk_pages = [chunks[chunk_index] for chunks in value_chunks]
        output += _mix_page_chunk_grouped_multiquery_mps(chunk_weights, chunk_pages, trace=trace)
        offset += chunk_token_count

    return logits, weights, output[:, :, :head_dim]


def decode_grouped_multiquery_step_prepared_mps_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageMPS]],
    value_pages_by_group: Sequence[Sequence[PreparedPageMPS]],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not key_pages_by_group or not value_pages_by_group:
        raise ValueError("grouped decode requires non-empty key/value page groups")
    if len(key_pages_by_group) != len(value_pages_by_group):
        raise ValueError("key_pages_by_group and value_pages_by_group must have the same group count")
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        raise ValueError("query_groups must align with key/value group count")
    if not key_pages_by_group[0]:
        raise ValueError("grouped decode requires at least one key page per group")

    query_tensors = [
        group.to(dtype=torch.float32, device="mps") if torch.is_tensor(group) else torch.as_tensor(group, dtype=torch.float32, device="mps")
        for group in query_groups
    ]
    query_count = int(query_tensors[0].shape[0])
    for group_query in query_tensors[1:]:
        if int(group_query.shape[0]) != query_count:
            raise ValueError("all query groups must have the same query count for batched grouped decode")

    first_key_group = key_pages_by_group[0]
    first_value_group = value_pages_by_group[0]
    key_chunks = _chunk_compatible_pages(first_key_group)
    value_chunks = _chunk_compatible_pages(first_value_group)
    if len(key_chunks) != len(value_chunks):
        raise ValueError("key and value chunk counts must align for grouped decode")

    key_chunk_lengths = [len(chunk) for chunk in key_chunks]
    value_chunk_lengths = [len(chunk) for chunk in value_chunks]

    logits_parts = []
    key_offset = 0
    for chunk_length in key_chunk_lengths:
        chunk_pages = [group_pages[key_offset : key_offset + chunk_length] for group_pages in key_pages_by_group]
        logits_parts.append(_score_page_chunk_grouped_multiquery_mps(query_tensors, chunk_pages, trace=trace))
        key_offset += chunk_length
    logits = torch.cat(logits_parts, dim=2)
    weights = torch.softmax(logits, dim=2)

    head_dim = first_value_group[0].header.head_dim
    padded_head_dim = first_value_group[0].header.padded_head_dim
    output = torch.zeros((group_count, query_count, padded_head_dim), dtype=torch.float32, device="mps")
    offset = 0
    value_offset = 0
    for chunk_index, chunk_length in enumerate(value_chunk_lengths):
        chunk_template = value_chunks[chunk_index]
        chunk_token_count = chunk_template[0].header.token_count * chunk_length
        chunk_weights = weights[:, :, offset : offset + chunk_token_count].reshape(
            group_count,
            query_count,
            chunk_length,
            chunk_template[0].header.token_count,
        )
        chunk_pages = [group_pages[value_offset : value_offset + chunk_length] for group_pages in value_pages_by_group]
        output += _mix_page_chunk_grouped_multiquery_mps(chunk_weights, chunk_pages, trace=trace)
        offset += chunk_token_count
        value_offset += chunk_length

    return logits, weights, output[:, :, :head_dim]


def decode_multi_query_step_mps(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageMPS],
    value_pages: Sequence[EncodedPage | PreparedPageMPS],
    *,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits, weights, output = decode_multi_query_step_mps_tensor(
        query_slices,
        key_pages,
        value_pages,
        trace=trace,
    )
    return (
        logits.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
    )
