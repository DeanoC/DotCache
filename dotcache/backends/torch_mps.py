from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tracing import ExecutionTrace
from ..types import EncodedPage, PageHeader


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


def prepare_page_mps(page: EncodedPage | PreparedPageMPS, *, trace: ExecutionTrace | None = None) -> PreparedPageMPS:
    if isinstance(page, PreparedPageMPS):
        return page
    if not mps_available():
        raise RuntimeError("torch_mps is unavailable on this machine")
    if not page_supported_mps(page):
        raise ValueError("page is unsupported by torch_mps in this phase")

    payload = None
    scales = None
    bias = None
    escape_payload = None
    unpack_shifts = None
    unpack_mask = None
    host_to_device_nbytes = 0

    if page.payload is not None:
        payload = _device_tensor(np.asarray(page.payload, dtype=np.int64), device="mps")
        host_to_device_nbytes += int(payload.numel() * payload.element_size())
    if page.scales is not None:
        scales = _device_tensor(np.asarray(page.scales, dtype=np.float32), device="mps")
        host_to_device_nbytes += int(scales.numel() * scales.element_size())
    if page.bias is not None:
        bias = _device_tensor(np.asarray(page.bias, dtype=np.float32), device="mps")
        host_to_device_nbytes += int(bias.numel() * bias.element_size())
    if page.escape_payload is not None:
        escape_payload = _device_tensor(np.asarray(page.escape_payload, dtype=np.float32), device="mps")
        host_to_device_nbytes += int(escape_payload.numel() * escape_payload.element_size())
    if page.header.mode_default == "M0":
        torch = _load_torch()
        symbols_per_word = 32 // page.header.bits
        unpack_shifts = torch.arange(symbols_per_word, dtype=torch.int64, device="mps") * page.header.bits
        unpack_mask = torch.tensor((1 << page.header.bits) - 1, dtype=torch.int64, device="mps")

    prepared = PreparedPageMPS(
        source_page=page,
        header=page.header,
        payload=payload,
        scales=scales,
        bias=bias,
        escape_payload=escape_payload,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
        host_to_device_nbytes=host_to_device_nbytes,
    )
    if trace is not None:
        trace.record_host_to_device(host_to_device_nbytes)
    return prepared


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


def _unpack_bits_torch(words, shifts, mask, group_size: int):
    torch = _load_torch()
    if words.ndim != 2:
        raise ValueError("words must have shape [token_count, words_per_group]")
    if shifts is None or mask is None:
        raise ValueError("prepared MPS pages require unpack metadata")
    expanded = torch.bitwise_and(torch.bitwise_right_shift(words[..., None], shifts), mask)
    return expanded.reshape(words.shape[0], group_size).to(torch.float32)


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
        dense = prepared.escape_payload[:, : header.head_dim]
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
        dense = prepared.escape_payload[:, : header.head_dim]
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
