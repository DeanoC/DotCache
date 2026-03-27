from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Literal, Sequence

import numpy as np
from ..tracing import ExecutionTrace
from ..types import EncodedPage, PageHeader
from ..modes.m2_key_sketch import segment_ids_for_token_count
from ..modes.turbo3 import TURBO3_CENTROIDS, fwht_last_dim
from ..packing import words_per_group

TorchDevice = Literal["mps", "cuda"]
PreparedDevice = Literal["torch_mps", "torch_cuda"]

_UNPACK_METADATA: dict[tuple[TorchDevice, int], tuple[Any, Any]] = {}
_SPILL_UNPACK_METADATA: dict[tuple[TorchDevice, int, int], tuple[Any, Any, Any, Any, Any, Any]] = {}
_TURBO3_CENTROID_TENSORS: dict[TorchDevice, Any] = {}
_FWHT_MATRICES: dict[tuple[TorchDevice, int], Any] = {}
_MAX_PREPARE_PAGES_PER_CHUNK = 128
_MPS_M0_KEY_PREPARE_PAGES_PER_CHUNK = 256
_MAX_PREPARED_CHUNK_CACHE_ENTRIES = 64
_MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES = 64 * 1024 * 1024
_MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT = 4
_PREPARED_CHUNK_CACHE_KINDS = frozenset({"K", "V"})
_PREPARED_CHUNK_CACHE: "OrderedDict[tuple[tuple[int, int], ...], PreparedChunkMPS]" = OrderedDict()
_PREPARED_CHUNK_CACHE_RESIDENT_BYTES = 0
_PREPARED_GROUPED_CHUNK_CACHE: "OrderedDict[tuple[tuple[tuple[int, int], ...], ...], PreparedGroupedChunkMPS]" = OrderedDict()
_PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES = 0
_PREPARED_PAGE_UID = 1


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in environments without torch
        raise RuntimeError("torch is required for the torch accelerator backends") from exc
    return torch


def _backend_name(device_type: TorchDevice) -> PreparedDevice:
    return "torch_cuda" if device_type == "cuda" else "torch_mps"


def torch_device_available(device_type: TorchDevice) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if device_type == "mps":
        return bool(torch.backends.mps.is_available())
    return bool(torch.cuda.is_available())


def mps_available() -> bool:
    return torch_device_available("mps")


def _device_tensor(array: np.ndarray, *, device: TorchDevice):
    torch = _load_torch()
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


@dataclass(slots=True)
class PreparedPageTorch:
    device_type: TorchDevice
    source_page: EncodedPage
    header: PageHeader
    payload: Any | None = None
    scales: Any | None = None
    bias: Any | None = None
    codebooks: Any | None = None
    m2_sketch: Any | None = None
    m2_basis: Any | None = None
    m2_mean: Any | None = None
    escape_payload: Any | None = None
    unpack_shifts: Any | None = None
    unpack_mask: Any | None = None
    host_to_device_nbytes: int = 0
    resident_nbytes: int = 0
    cache_uid: int = 0

    @property
    def payload_nbytes(self) -> int:
        return self.source_page.payload_nbytes

    @property
    def metadata_nbytes(self) -> int:
        return self.source_page.metadata_nbytes


PreparedPageMPS = PreparedPageTorch


@dataclass(slots=True)
class PreparedChunkMPS:
    header: PageHeader
    payload_groups: tuple[Any, ...]
    codes_groups: tuple[Any, ...] | None
    scales_groups: tuple[Any, ...] | None
    bias_groups: tuple[Any, ...] | None
    resident_nbytes: int


@dataclass(slots=True)
class PreparedGroupedChunkMPS:
    header: PageHeader
    payload_groups: tuple[Any, ...]
    codes_groups: tuple[Any, ...] | None
    scales_groups: tuple[Any, ...] | None
    bias_groups: tuple[Any, ...] | None
    resident_nbytes: int


def prepared_chunk_cache_resident_bytes() -> int:
    return int(_PREPARED_CHUNK_CACHE_RESIDENT_BYTES + _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES)


def configure_prepared_chunk_cache(
    *,
    max_entries: int | None = None,
    max_resident_bytes: int | None = None,
    min_page_count: int | None = None,
    cached_kinds: Sequence[str] | None = None,
    clear: bool = True,
) -> None:
    global _MAX_PREPARED_CHUNK_CACHE_ENTRIES
    global _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    global _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT
    global _PREPARED_CHUNK_CACHE_KINDS
    if max_entries is not None:
        _MAX_PREPARED_CHUNK_CACHE_ENTRIES = max(0, int(max_entries))
    if max_resident_bytes is not None:
        _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES = max(0, int(max_resident_bytes))
    if min_page_count is not None:
        _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT = max(1, int(min_page_count))
    if cached_kinds is not None:
        _PREPARED_CHUNK_CACHE_KINDS = frozenset(str(kind) for kind in cached_kinds)
    if clear:
        clear_prepared_chunk_cache()


def clear_prepared_chunk_cache() -> None:
    global _PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    global _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES
    _PREPARED_CHUNK_CACHE.clear()
    _PREPARED_CHUNK_CACHE_RESIDENT_BYTES = 0
    _PREPARED_GROUPED_CHUNK_CACHE.clear()
    _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES = 0


def _next_prepared_page_uid() -> int:
    global _PREPARED_PAGE_UID
    page_uid = _PREPARED_PAGE_UID
    _PREPARED_PAGE_UID += 1
    return page_uid


def _prepare_signature(page: EncodedPage | PreparedPageTorch) -> tuple[int | str, ...]:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    header = source_page.header
    sketch_dim = int(source_page.m2_sketch.shape[-1]) if source_page.m2_sketch is not None else 0
    segment_count = int(source_page.m2_basis.shape[1]) if source_page.m2_basis is not None and source_page.m2_basis.ndim == 4 else 1
    centered = int(source_page.m2_mean is not None)
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
        sketch_dim,
        segment_count,
        centered,
    )


def _max_prepare_pages_for_source_page(page: EncodedPage | PreparedPageTorch, *, device_type: TorchDevice) -> int:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    if device_type == "mps" and source_page.header.mode_default == "M0" and source_page.header.kind == "K":
        return _MPS_M0_KEY_PREPARE_PAGES_PER_CHUNK
    return _MAX_PREPARE_PAGES_PER_CHUNK


def _batched_signature(page: PreparedPageTorch) -> tuple[int | str, ...]:
    header = page.header
    sketch_dim = int(page.m2_sketch.shape[-1]) if page.m2_sketch is not None else 0
    segment_count = int(page.m2_basis.shape[1]) if page.m2_basis is not None and page.m2_basis.dim() == 4 else 1
    centered = int(page.m2_mean is not None)
    return (
        page.device_type,
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
        sketch_dim,
        segment_count,
        centered,
    )


def _chunk_compatible_source_pages(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
) -> list[list[EncodedPage | PreparedPageTorch]]:
    chunks: list[list[EncodedPage | PreparedPageTorch]] = []
    current_chunk: list[EncodedPage | PreparedPageTorch] = []
    current_signature: tuple[int | str, ...] | None = None
    current_limit = _MAX_PREPARE_PAGES_PER_CHUNK
    for page in pages:
        signature = _prepare_signature(page)
        if current_chunk and (
            signature != current_signature or len(current_chunk) >= current_limit
        ):
            chunks.append(current_chunk)
            current_chunk = [page]
            current_signature = signature
            current_limit = _max_prepare_pages_for_source_page(page, device_type=device_type)
            continue
        if not current_chunk:
            current_signature = signature
            current_limit = _max_prepare_pages_for_source_page(page, device_type=device_type)
        current_chunk.append(page)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _chunk_compatible_pages(pages: Sequence[PreparedPageTorch]) -> list[list[PreparedPageTorch]]:
    chunks: list[list[PreparedPageTorch]] = []
    current_chunk: list[PreparedPageTorch] = []
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


def _prepared_chunk_cache_key(pages: Sequence[PreparedPageTorch]) -> tuple[tuple[int, int], ...] | None:
    if not pages:
        return None
    if pages[0].header.mode_default not in ("M0", "T3"):
        return None
    return tuple((int(page.cache_uid), int(page.header.token_count)) for page in pages)


def _build_prepared_chunk_mps(pages: Sequence[PreparedPageTorch]) -> PreparedChunkMPS:
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    if header.mode_default not in ("M0", "T3"):
        raise ValueError("prepared chunk cache currently supports only M0 and T3 pages")
    payload_groups = tuple(torch.stack([page.payload[group_index] for page in pages], dim=0) for group_index in range(header.num_groups))
    codes_groups = tuple(
        _unpack_bits_torch(
            payload_groups[group_index].reshape(-1, header.words_per_group),
            pages[0].unpack_shifts,
            pages[0].unpack_mask,
            header.group_size,
        ).reshape(len(pages), header.token_count, header.group_size)
        for group_index in range(header.num_groups)
    )
    scales_groups = tuple(
        torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        for group_index in range(header.num_groups)
    )
    bias_groups = (
        tuple(
            torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
            for group_index in range(header.num_groups)
        )
        if header.mode_default == "M0"
        else None
    )
    resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in payload_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in codes_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in scales_groups)
    if bias_groups is not None:
        resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups)
    return PreparedChunkMPS(
        header=header,
        payload_groups=payload_groups,
        codes_groups=codes_groups,
        scales_groups=scales_groups,
        bias_groups=bias_groups,
        resident_nbytes=resident_nbytes,
    )


def _get_prepared_chunk_mps(pages: Sequence[PreparedPageTorch]) -> PreparedChunkMPS | None:
    global _PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    cache_key = _prepared_chunk_cache_key(pages)
    if cache_key is None:
        return None
    if pages[0].header.kind not in _PREPARED_CHUNK_CACHE_KINDS:
        return None
    if len(pages) < _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT:
        return None
    cached_chunk = _PREPARED_CHUNK_CACHE.get(cache_key)
    if cached_chunk is not None:
        _PREPARED_CHUNK_CACHE.move_to_end(cache_key)
        return cached_chunk
    prepared_chunk = _build_prepared_chunk_mps(pages)
    if (
        _MAX_PREPARED_CHUNK_CACHE_ENTRIES <= 0
        or _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES <= 0
        or prepared_chunk.resident_nbytes > _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    ):
        return prepared_chunk
    _PREPARED_CHUNK_CACHE[cache_key] = prepared_chunk
    _PREPARED_CHUNK_CACHE_RESIDENT_BYTES += prepared_chunk.resident_nbytes
    while (
        len(_PREPARED_CHUNK_CACHE) > _MAX_PREPARED_CHUNK_CACHE_ENTRIES
        or _PREPARED_CHUNK_CACHE_RESIDENT_BYTES > _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    ):
        _, evicted_chunk = _PREPARED_CHUNK_CACHE.popitem(last=False)
        _PREPARED_CHUNK_CACHE_RESIDENT_BYTES = max(
            0,
            _PREPARED_CHUNK_CACHE_RESIDENT_BYTES - evicted_chunk.resident_nbytes,
        )
    return prepared_chunk


def _grouped_prepared_chunk_cache_key(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    if not pages_by_group or not pages_by_group[0]:
        return None
    if pages_by_group[0][0].header.mode_default != "M0":
        return None
    page_count = len(pages_by_group[0])
    cache_key: list[tuple[tuple[int, int], ...]] = []
    for group_pages in pages_by_group:
        if len(group_pages) != page_count:
            return None
        group_key = _prepared_chunk_cache_key(group_pages)
        if group_key is None:
            return None
        cache_key.append(group_key)
    return tuple(cache_key)


def _build_grouped_prepared_chunk_mps(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> PreparedGroupedChunkMPS | None:
    if not pages_by_group or not pages_by_group[0]:
        return None
    prepared_chunks = [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
    if any(chunk is None for chunk in prepared_chunks):
        return None
    header = pages_by_group[0][0].header
    payload_groups = tuple(
        _load_torch().stack([chunk.payload_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    codes_groups = tuple(
        _load_torch().stack([chunk.codes_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    scales_groups = tuple(
        _load_torch().stack([chunk.scales_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    bias_groups = tuple(
        _load_torch().stack([chunk.bias_groups[group_index] for chunk in prepared_chunks], dim=0)
        for group_index in range(header.num_groups)
    )
    resident_nbytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in payload_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in codes_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in scales_groups)
    resident_nbytes += sum(int(tensor.numel() * tensor.element_size()) for tensor in bias_groups)
    return PreparedGroupedChunkMPS(
        header=header,
        payload_groups=payload_groups,
        codes_groups=codes_groups,
        scales_groups=scales_groups,
        bias_groups=bias_groups,
        resident_nbytes=resident_nbytes,
    )


def _get_grouped_prepared_chunk_mps(
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
) -> PreparedGroupedChunkMPS | None:
    global _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES
    cache_key = _grouped_prepared_chunk_cache_key(pages_by_group)
    if cache_key is None:
        return None
    if pages_by_group[0][0].header.kind not in _PREPARED_CHUNK_CACHE_KINDS:
        return None
    if len(pages_by_group[0]) < _MIN_PREPARED_CHUNK_CACHE_PAGE_COUNT:
        return None
    cached_chunk = _PREPARED_GROUPED_CHUNK_CACHE.get(cache_key)
    if cached_chunk is not None:
        _PREPARED_GROUPED_CHUNK_CACHE.move_to_end(cache_key)
        return cached_chunk
    prepared_chunk = _build_grouped_prepared_chunk_mps(pages_by_group)
    if prepared_chunk is None:
        return None
    if (
        _MAX_PREPARED_CHUNK_CACHE_ENTRIES <= 0
        or _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES <= 0
        or prepared_chunk.resident_nbytes > _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    ):
        return prepared_chunk
    _PREPARED_GROUPED_CHUNK_CACHE[cache_key] = prepared_chunk
    _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES += prepared_chunk.resident_nbytes
    while (
        len(_PREPARED_GROUPED_CHUNK_CACHE) > _MAX_PREPARED_CHUNK_CACHE_ENTRIES
        or _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES > _MAX_PREPARED_CHUNK_CACHE_RESIDENT_BYTES
    ):
        _, evicted_chunk = _PREPARED_GROUPED_CHUNK_CACHE.popitem(last=False)
        _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES = max(
            0,
            _PREPARED_GROUPED_CHUNK_CACHE_RESIDENT_BYTES - evicted_chunk.resident_nbytes,
        )
    return prepared_chunk


def page_supported_torch(page: EncodedPage | PreparedPageTorch) -> bool:
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    header = source_page.header
    if header.layout != "group_major":
        return False
    if header.mode_default == "M3":
        return source_page.escape_payload is not None
    if header.mode_default == "M2":
        return (
            header.kind == "K"
            and header.quant_scheme == "sketch"
            and source_page.m2_sketch is not None
            and source_page.m2_basis is not None
            and source_page.m2_mean is not None
        )
    if header.mode_default == "T3":
        return (
            header.quant_scheme == "turbo3"
            and header.bits == 3
            and header.group_size in (32, 64)
            and source_page.payload is not None
            and source_page.scales is not None
            and source_page.codebooks is not None
        )
    return (
        header.mode_default in ("M0", "M1")
        and header.bits in (2, 4)
        and header.group_size in (32, 64)
        and source_page.payload is not None
        and (
            (
                header.mode_default == "M0"
                and header.quant_scheme == "affine"
                and source_page.scales is not None
                and source_page.bias is not None
            )
            or (
                header.mode_default == "M1"
                and header.quant_scheme == "lut"
                and source_page.codebooks is not None
            )
        )
    )


def page_supported_mps(page: EncodedPage | PreparedPageTorch) -> bool:
    return page_supported_torch(page)


def _unpack_metadata(bits: int, *, device_type: TorchDevice):
    cache_key = (device_type, bits)
    cached = _UNPACK_METADATA.get(cache_key)
    if cached is not None:
        return cached
    torch = _load_torch()
    symbols_per_word = 32 // bits
    shifts = torch.arange(symbols_per_word, dtype=torch.int32, device=device_type) * bits
    mask = torch.tensor((1 << bits) - 1, dtype=torch.int32, device=device_type)
    _UNPACK_METADATA[cache_key] = (shifts, mask)
    return shifts, mask


def _spill_unpack_metadata(bits: int, group_size: int, *, device_type: TorchDevice):
    cache_key = (device_type, bits, group_size)
    cached = _SPILL_UNPACK_METADATA.get(cache_key)
    if cached is not None:
        return cached
    torch = _load_torch()
    bit_offsets = np.arange(group_size, dtype=np.int64) * int(bits)
    word_count = words_per_group(group_size, bits)
    word_indices = torch.as_tensor(bit_offsets // 32, dtype=torch.int64, device=device_type)
    next_word_indices = torch.as_tensor(
        np.minimum(bit_offsets // 32 + 1, word_count - 1),
        dtype=torch.int64,
        device=device_type,
    )
    bit_indices = torch.as_tensor(bit_offsets % 32, dtype=torch.int64, device=device_type)
    spill_width = np.maximum((bit_offsets % 32) + int(bits) - 32, 0).astype(np.int64)
    spill_mask = torch.as_tensor((1 << spill_width) - 1, dtype=torch.int64, device=device_type)
    shift_back = torch.as_tensor(int(bits) - spill_width, dtype=torch.int64, device=device_type)
    spill_flags = torch.as_tensor(spill_width > 0, dtype=torch.bool, device=device_type)
    _SPILL_UNPACK_METADATA[cache_key] = (
        word_indices,
        next_word_indices,
        bit_indices,
        spill_mask,
        shift_back,
        spill_flags,
    )
    return _SPILL_UNPACK_METADATA[cache_key]


def _turbo3_centroids_torch(*, device_type: TorchDevice):
    cached = _TURBO3_CENTROID_TENSORS.get(device_type)
    if cached is not None:
        return cached
    tensor = _device_tensor(TURBO3_CENTROIDS.astype(np.float32, copy=False), device=device_type)
    tensor = tensor.to(dtype=_load_torch().float32)
    _TURBO3_CENTROID_TENSORS[device_type] = tensor
    return tensor


def _fwht_matrix_torch(width: int, *, device_type: TorchDevice):
    cache_key = (device_type, int(width))
    cached = _FWHT_MATRICES.get(cache_key)
    if cached is not None:
        return cached
    if width <= 0 or (width & (width - 1)):
        raise ValueError("FWHT requires the last dimension to be a power of two")
    basis = np.eye(width, dtype=np.float32)
    transformed = fwht_last_dim(basis)
    tensor = _device_tensor(transformed, device=device_type).to(dtype=_load_torch().float32)
    _FWHT_MATRICES[cache_key] = tensor
    return tensor


def _synchronize_torch_device(device_type: TorchDevice) -> None:
    torch = _load_torch()
    if device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        return
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _trace_timed_call(
    trace: ExecutionTrace | None,
    section: str,
    *,
    device_type: TorchDevice,
    fn,
    synchronize: bool = True,
):
    if trace is None or not trace.capture_timings:
        return fn()
    if synchronize:
        _synchronize_torch_device(device_type)
    start = time.perf_counter()
    result = fn()
    if synchronize:
        _synchronize_torch_device(device_type)
    trace.record_timing(section, (time.perf_counter() - start) * 1000.0)
    return result


def _prepared_page_host_nbytes(page: EncodedPage) -> int:
    total = 0
    if page.payload is not None:
        total += int(page.payload.nbytes)
    if page.scales is not None:
        total += int(page.scales.nbytes)
    if page.bias is not None:
        total += int(page.bias.nbytes)
    if page.codebooks is not None:
        total += int(page.codebooks.nbytes)
    if page.m2_sketch is not None:
        total += int(page.m2_sketch.nbytes)
    if page.m2_basis is not None:
        total += int(page.m2_basis.nbytes)
    if page.m2_mean is not None:
        total += int(page.m2_mean.nbytes)
    if page.escape_payload is not None:
        total += int(page.escape_payload.nbytes)
    return total


def _optional_m2_sidecar_batches(
    pages: Sequence[EncodedPage],
    *,
    device_type: TorchDevice,
) -> tuple[Any | None, Any | None, Any | None, int, int]:
    if not pages or not all(page.m2_sketch is not None and page.m2_basis is not None and page.m2_mean is not None for page in pages):
        return None, None, None, 0, 0
    sketch_array = np.stack([np.asarray(page.m2_sketch) for page in pages], axis=0)
    basis_array = np.stack([np.asarray(page.m2_basis) for page in pages], axis=0)
    mean_array = np.stack([np.asarray(page.m2_mean) for page in pages], axis=0)
    sketch_batch = _device_tensor(sketch_array, device=device_type)
    basis_batch = _device_tensor(basis_array, device=device_type)
    mean_batch = _device_tensor(mean_array, device=device_type)
    if device_type == "mps":
        sketch_batch = sketch_batch.to(dtype=_load_torch().float32)
        basis_batch = basis_batch.to(dtype=_load_torch().float32)
        mean_batch = mean_batch.to(dtype=_load_torch().float32)
    return sketch_batch, basis_batch, mean_batch, int(sketch_array.nbytes + basis_array.nbytes + mean_array.nbytes), int(
        sketch_batch.numel() * sketch_batch.element_size()
        + basis_batch.numel() * basis_batch.element_size()
        + mean_batch.numel() * mean_batch.element_size()
    )


def _prepare_page_chunk_torch(
    pages: Sequence[EncodedPage],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    if not pages:
        return []
    header = pages[0].header
    total_host_to_device_nbytes = 0

    if header.mode_default == "M3":
        escape_batch = _device_tensor(np.stack([np.asarray(page.escape_payload) for page in pages], axis=0), device=device_type)
        total_host_to_device_nbytes += int(escape_batch.numel() * escape_batch.element_size())
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                escape_payload=escape_batch[index],
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=int(escape_batch[index].numel() * escape_batch[index].element_size()),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "M2":
        sketch_array = np.stack([np.asarray(page.m2_sketch) for page in pages], axis=0)
        basis_array = np.stack([np.asarray(page.m2_basis) for page in pages], axis=0)
        mean_array = np.stack([np.asarray(page.m2_mean) for page in pages], axis=0)
        sketch_batch = _device_tensor(sketch_array, device=device_type)
        basis_batch = _device_tensor(basis_array, device=device_type)
        mean_batch = _device_tensor(mean_array, device=device_type)
        total_host_to_device_nbytes += int(sketch_array.nbytes)
        total_host_to_device_nbytes += int(basis_array.nbytes)
        total_host_to_device_nbytes += int(mean_array.nbytes)
        if device_type == "mps":
            sketch_batch = sketch_batch.to(dtype=_load_torch().float32)
            basis_batch = basis_batch.to(dtype=_load_torch().float32)
            mean_batch = mean_batch.to(dtype=_load_torch().float32)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                m2_sketch=sketch_batch[index],
                m2_basis=basis_batch[index],
                m2_mean=mean_batch[index],
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(sketch_batch[index].numel() * sketch_batch[index].element_size())
                    + int(basis_batch[index].numel() * basis_batch[index].element_size())
                    + int(mean_batch[index].numel() * mean_batch[index].element_size())
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "M1":
        payload_array = np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0)
        codebooks_array = np.stack([np.asarray(page.codebooks) for page in pages], axis=0)
        payload_batch = _device_tensor(payload_array, device=device_type)
        codebooks_batch = _device_tensor(codebooks_array, device=device_type)
        sidecar_sketch_batch, sidecar_basis_batch, sidecar_mean_batch, sidecar_h2d_nbytes, _ = _optional_m2_sidecar_batches(
            pages,
            device_type=device_type,
        )
        total_host_to_device_nbytes += int(payload_array.nbytes)
        total_host_to_device_nbytes += int(codebooks_array.nbytes)
        total_host_to_device_nbytes += sidecar_h2d_nbytes
        if device_type == "mps":
            codebooks_batch = codebooks_batch.to(dtype=_load_torch().float32)
        unpack_shifts, unpack_mask = _unpack_metadata(header.bits, device_type=device_type)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                payload=payload_batch[index],
                codebooks=codebooks_batch[index],
                m2_sketch=None if sidecar_sketch_batch is None else sidecar_sketch_batch[index],
                m2_basis=None if sidecar_basis_batch is None else sidecar_basis_batch[index],
                m2_mean=None if sidecar_mean_batch is None else sidecar_mean_batch[index],
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(payload_batch[index].numel() * payload_batch[index].element_size())
                    + int(codebooks_batch[index].numel() * codebooks_batch[index].element_size())
                    + (
                        0
                        if sidecar_sketch_batch is None or sidecar_basis_batch is None or sidecar_mean_batch is None
                        else int(sidecar_sketch_batch[index].numel() * sidecar_sketch_batch[index].element_size())
                        + int(sidecar_basis_batch[index].numel() * sidecar_basis_batch[index].element_size())
                        + int(sidecar_mean_batch[index].numel() * sidecar_mean_batch[index].element_size())
                    )
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    if header.mode_default == "T3":
        payload_array = np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0)
        scales_array = np.stack([np.asarray(page.scales) for page in pages], axis=0)
        payload_batch = _device_tensor(payload_array, device=device_type)
        scales_batch = _device_tensor(scales_array, device=device_type)
        sidecar_sketch_batch, sidecar_basis_batch, sidecar_mean_batch, sidecar_h2d_nbytes, _ = _optional_m2_sidecar_batches(
            pages,
            device_type=device_type,
        )
        total_host_to_device_nbytes += int(payload_array.nbytes)
        total_host_to_device_nbytes += int(scales_array.nbytes)
        total_host_to_device_nbytes += sidecar_h2d_nbytes
        if device_type == "mps":
            scales_batch = scales_batch.to(dtype=_load_torch().float32)
        codebooks_tensor = _turbo3_centroids_torch(device_type=device_type)
        unpack_shifts, unpack_mask = _unpack_metadata(header.bits, device_type=device_type)
        prepared_pages = [
            PreparedPageTorch(
                device_type=device_type,
                source_page=page,
                header=page.header,
                payload=payload_batch[index],
                scales=scales_batch[index],
                codebooks=codebooks_tensor,
                m2_sketch=None if sidecar_sketch_batch is None else sidecar_sketch_batch[index],
                m2_basis=None if sidecar_basis_batch is None else sidecar_basis_batch[index],
                m2_mean=None if sidecar_mean_batch is None else sidecar_mean_batch[index],
                unpack_shifts=unpack_shifts,
                unpack_mask=unpack_mask,
                host_to_device_nbytes=_prepared_page_host_nbytes(page),
                resident_nbytes=(
                    int(payload_batch[index].numel() * payload_batch[index].element_size())
                    + int(scales_batch[index].numel() * scales_batch[index].element_size())
                    + (
                        0
                        if sidecar_sketch_batch is None or sidecar_basis_batch is None or sidecar_mean_batch is None
                        else int(sidecar_sketch_batch[index].numel() * sidecar_sketch_batch[index].element_size())
                        + int(sidecar_basis_batch[index].numel() * sidecar_basis_batch[index].element_size())
                        + int(sidecar_mean_batch[index].numel() * sidecar_mean_batch[index].element_size())
                    )
                ),
                cache_uid=_next_prepared_page_uid(),
            )
            for index, page in enumerate(pages)
        ]
        if trace is not None:
            trace.record_host_to_device(total_host_to_device_nbytes)
        return prepared_pages

    payload_array = np.stack([np.asarray(page.payload, dtype=np.int32) for page in pages], axis=0)
    scales_array = np.stack([np.asarray(page.scales) for page in pages], axis=0)
    bias_array = np.stack([np.asarray(page.bias) for page in pages], axis=0)
    payload_batch = _device_tensor(payload_array, device=device_type)
    scales_batch = _device_tensor(scales_array, device=device_type)
    bias_batch = _device_tensor(bias_array, device=device_type)
    sidecar_sketch_batch, sidecar_basis_batch, sidecar_mean_batch, sidecar_h2d_nbytes, _ = _optional_m2_sidecar_batches(
        pages,
        device_type=device_type,
    )
    total_host_to_device_nbytes += int(payload_array.nbytes)
    total_host_to_device_nbytes += int(scales_array.nbytes)
    total_host_to_device_nbytes += int(bias_array.nbytes)
    total_host_to_device_nbytes += sidecar_h2d_nbytes
    if device_type == "mps":
        scales_batch = scales_batch.to(dtype=_load_torch().float32)
        bias_batch = bias_batch.to(dtype=_load_torch().float32)
    unpack_shifts, unpack_mask = _unpack_metadata(header.bits, device_type=device_type)

    prepared_pages = [
        PreparedPageTorch(
            device_type=device_type,
            source_page=page,
            header=page.header,
            payload=payload_batch[index],
            scales=scales_batch[index],
            bias=bias_batch[index],
            m2_sketch=None if sidecar_sketch_batch is None else sidecar_sketch_batch[index],
            m2_basis=None if sidecar_basis_batch is None else sidecar_basis_batch[index],
            m2_mean=None if sidecar_mean_batch is None else sidecar_mean_batch[index],
            unpack_shifts=unpack_shifts,
            unpack_mask=unpack_mask,
            host_to_device_nbytes=_prepared_page_host_nbytes(page),
            resident_nbytes=(
                int(payload_batch[index].numel() * payload_batch[index].element_size())
                + int(scales_batch[index].numel() * scales_batch[index].element_size())
                + int(bias_batch[index].numel() * bias_batch[index].element_size())
                + (
                    0
                    if sidecar_sketch_batch is None or sidecar_basis_batch is None or sidecar_mean_batch is None
                    else int(sidecar_sketch_batch[index].numel() * sidecar_sketch_batch[index].element_size())
                    + int(sidecar_basis_batch[index].numel() * sidecar_basis_batch[index].element_size())
                    + int(sidecar_mean_batch[index].numel() * sidecar_mean_batch[index].element_size())
                )
            ),
            cache_uid=_next_prepared_page_uid(),
        )
        for index, page in enumerate(pages)
    ]
    if trace is not None:
        trace.record_host_to_device(total_host_to_device_nbytes)
    return prepared_pages


def prepare_pages_torch(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    backend_name = _backend_name(device_type)
    if not torch_device_available(device_type):
        raise RuntimeError(f"{backend_name} is unavailable on this machine")
    prepared_pages: list[PreparedPageTorch] = []
    for page_chunk in _chunk_compatible_source_pages(pages, device_type=device_type):
        if all(isinstance(page, PreparedPageTorch) and page.device_type == device_type for page in page_chunk):
            prepared_pages.extend(page_chunk)  # type: ignore[arg-type]
            continue
        source_pages = []
        for page in page_chunk:
            source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
            if not page_supported_torch(source_page):
                raise ValueError(f"page is unsupported by {backend_name} in this phase")
            source_pages.append(source_page)
        if source_pages:
            prepared_pages.extend(
                _trace_timed_call(
                    trace,
                    "prepare",
                    device_type=device_type,
                    fn=lambda source_pages=source_pages: _prepare_page_chunk_torch(
                        source_pages,
                        device_type=device_type,
                        trace=trace,
                    ),
                )
            )
    return prepared_pages


def prepare_page_torch(
    page: EncodedPage | PreparedPageTorch,
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> PreparedPageTorch:
    if isinstance(page, PreparedPageTorch) and page.device_type == device_type:
        return page
    source_page = page.source_page if isinstance(page, PreparedPageTorch) else page
    return prepare_pages_torch([source_page], device_type=device_type, trace=trace)[0]


def _pad_query(query_slice: np.ndarray | Any, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    if torch.is_tensor(query_slice):
        query = query_slice.to(dtype=torch.float32, device=device_type)
    else:
        query = torch.as_tensor(query_slice, dtype=torch.float32, device=device_type)
    if query.ndim != 1:
        raise ValueError("query_slice must have shape [head_dim]")
    if int(query.shape[0]) > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if int(query.shape[0]) == padded_head_dim:
        return query
    padded = torch.zeros(padded_head_dim, dtype=torch.float32, device=device_type)
    padded[: query.shape[0]] = query
    return padded


def _pad_queries(query_slices: np.ndarray | Any, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    if torch.is_tensor(query_slices):
        queries = query_slices.to(dtype=torch.float32, device=device_type)
    else:
        queries = torch.as_tensor(query_slices, dtype=torch.float32, device=device_type)
    if queries.ndim != 2:
        raise ValueError("query_slices must have shape [query_count, head_dim]")
    if int(queries.shape[1]) > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if int(queries.shape[1]) == padded_head_dim:
        return queries
    padded = torch.zeros((queries.shape[0], padded_head_dim), dtype=torch.float32, device=device_type)
    padded[:, : queries.shape[1]] = queries
    return padded


def _prepare_output_accumulator(out_acc: np.ndarray | None, head_dim: int, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    output = torch.zeros(padded_head_dim, dtype=torch.float32, device=device_type)
    if out_acc is None:
        return output
    values = torch.as_tensor(out_acc, dtype=torch.float32, device=device_type)
    if values.shape != (head_dim,):
        raise ValueError("out_acc must have shape [head_dim]")
    output[:head_dim] = values
    return output


def _prepare_output_accumulator_tensor(out_acc, head_dim: int, padded_head_dim: int, *, device_type: TorchDevice):
    torch = _load_torch()
    if out_acc is None:
        return torch.zeros(padded_head_dim, dtype=torch.float32, device=device_type)
    if isinstance(out_acc, np.ndarray):
        return _prepare_output_accumulator(out_acc, head_dim, padded_head_dim, device_type=device_type)
    if tuple(out_acc.shape) != (padded_head_dim,):
        raise ValueError("out_acc tensor must have shape [padded_head_dim]")
    return out_acc.to(dtype=torch.float32, device=device_type)


def _unpack_bits_torch(words, shifts, mask, group_size: int, *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    if words.ndim != 2:
        raise ValueError("words must have shape [token_count, words_per_group]")
    if shifts is None or mask is None:
        raise ValueError("prepared torch pages require unpack metadata")
    device_type = str(words.device.type)

    def _impl():
        words_u64 = torch.bitwise_and(words.to(dtype=torch.int64), 0xFFFFFFFF)
        mask_i64 = torch.as_tensor(mask, dtype=torch.int64, device=words.device)
        bits = int(mask_i64.item()).bit_length()
        if 32 % bits == 0:
            shifts_i64 = shifts.to(dtype=torch.int64)
            expanded = torch.bitwise_and(torch.bitwise_right_shift(words_u64[..., None], shifts_i64), mask_i64)
            return expanded.reshape(words.shape[0], -1)[:, :group_size].to(torch.float32)
        if words.device.type == "mps":
            word_indices, next_word_indices, bit_indices, spill_mask, shift_back, spill_flags = _spill_unpack_metadata(
                bits,
                group_size,
                device_type=words.device.type,
            )
            bit_indices_2d = bit_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
            spill_mask_2d = spill_mask.unsqueeze(0).expand(words_u64.shape[0], -1)
            shift_back_2d = shift_back.unsqueeze(0).expand(words_u64.shape[0], -1)
            current_words = words_u64[:, word_indices]
            next_words = words_u64[:, next_word_indices]
            values = torch.bitwise_right_shift(current_words, bit_indices_2d)
            spilled = torch.bitwise_left_shift(torch.bitwise_and(next_words, spill_mask_2d), shift_back_2d)
            values = torch.where(spill_flags.unsqueeze(0), torch.bitwise_or(values, spilled), values)
            return torch.bitwise_and(values, mask_i64).to(torch.float32)
        word_indices, next_word_indices, bit_indices, spill_mask, shift_back, spill_flags = _spill_unpack_metadata(
            bits,
            group_size,
            device_type=words.device.type,
        )
        gather_index = word_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
        bit_indices_2d = bit_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
        values = torch.bitwise_right_shift(torch.gather(words_u64, 1, gather_index), bit_indices_2d)
        if bool(spill_flags.any()):
            next_gather_index = next_word_indices.unsqueeze(0).expand(words_u64.shape[0], -1)
            spill_mask_2d = spill_mask.unsqueeze(0).expand(words_u64.shape[0], -1)
            shift_back_2d = shift_back.unsqueeze(0).expand(words_u64.shape[0], -1)
            spilled = torch.bitwise_left_shift(
                torch.bitwise_and(torch.gather(words_u64, 1, next_gather_index), spill_mask_2d),
                shift_back_2d,
            )
            values = torch.where(spill_flags.unsqueeze(0), torch.bitwise_or(values, spilled), values)
        return torch.bitwise_and(values, mask_i64).to(torch.float32)

    return _trace_timed_call(trace, "unpack", device_type=device_type, fn=_impl, synchronize=False)


def _lookup_lut_group_torch(codebooks, codes):
    torch = _load_torch()
    lut = codebooks.to(dtype=torch.float32)
    code_indices = codes.to(dtype=torch.int64)
    if lut.ndim == 1 and code_indices.ndim == 1:
        return lut[code_indices]
    if lut.ndim == 2 and code_indices.ndim == 1 and lut.shape[0] == code_indices.shape[0]:
        page_index = torch.arange(lut.shape[0], device=lut.device)
        return lut[page_index, code_indices]
    if lut.ndim == 2 and code_indices.ndim == 2:
        token_count = int(code_indices.shape[0])
        segment_count = int(lut.shape[0])
        if segment_count == 1:
            return lut[0][code_indices]
        segment_ids = (torch.arange(token_count, device=lut.device, dtype=torch.int64) * segment_count) // max(token_count, 1)
        return lut[segment_ids[:, None], code_indices]
    if lut.ndim == 3 and code_indices.ndim == 3:
        page_count = int(code_indices.shape[0])
        token_count = int(code_indices.shape[1])
        segment_count = int(lut.shape[1])
        if segment_count == 1:
            page_index = torch.arange(page_count, device=lut.device)[:, None, None]
            return lut[page_index, torch.zeros(1, device=lut.device, dtype=torch.int64), code_indices]
        page_index = torch.arange(page_count, device=lut.device)[:, None, None]
        segment_ids = (torch.arange(token_count, device=lut.device, dtype=torch.int64) * segment_count) // max(token_count, 1)
        return lut[page_index, segment_ids[None, :, None], code_indices]
    if lut.ndim == 4 and code_indices.ndim == 4:
        batch_size = int(code_indices.shape[0])
        page_count = int(code_indices.shape[1])
        token_count = int(code_indices.shape[2])
        segment_count = int(lut.shape[2])
        batch_index = torch.arange(batch_size, device=lut.device)[:, None, None, None]
        page_index = torch.arange(page_count, device=lut.device)[None, :, None, None]
        if segment_count == 1:
            return lut[batch_index, page_index, torch.zeros(1, device=lut.device, dtype=torch.int64), code_indices]
        segment_ids = (torch.arange(token_count, device=lut.device, dtype=torch.int64) * segment_count) // max(token_count, 1)
        return lut[batch_index, page_index, segment_ids[None, None, :, None], code_indices]
    raise ValueError("unsupported LUT rank")


def _lookup_turbo_group_torch(codebooks, codes):
    torch = _load_torch()
    lut = codebooks.to(dtype=torch.float32)
    code_indices = codes.to(dtype=torch.int64)
    if lut.ndim == 1 and code_indices.ndim == 2:
        return lut[code_indices]
    if lut.ndim == 1 and code_indices.ndim == 3:
        return lut[code_indices]
    if lut.ndim == 1 and code_indices.ndim == 4:
        return lut[code_indices]
    if lut.ndim == 2 and code_indices.ndim == 3 and lut.shape[0] == code_indices.shape[0]:
        page_index = torch.arange(lut.shape[0], device=lut.device)[:, None, None]
        return lut[page_index, code_indices]
    if lut.ndim == 3 and code_indices.ndim == 4:
        batch_index = torch.arange(lut.shape[0], device=lut.device)[:, None, None, None]
        page_index = torch.arange(lut.shape[1], device=lut.device)[None, :, None, None]
        return lut[batch_index, page_index, code_indices]
    raise ValueError("unsupported turbo3 LUT rank")


def _fwht_last_dim_torch(values, *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    width = int(values.shape[-1])
    if width <= 0:
        return values
    if width & (width - 1):
        raise ValueError("FWHT requires the last dimension to be a power of two")
    device_type = str(values.device.type)

    def _impl():
        original_shape = tuple(values.shape)
        matrix = _fwht_matrix_torch(width, device_type=device_type)
        transformed = values.to(dtype=torch.float32).reshape(-1, width)
        return torch.matmul(transformed, matrix.T).reshape(original_shape)

    return _trace_timed_call(trace, "fwht", device_type=device_type, fn=_impl, synchronize=False)


def _score_page_chunk_torch(query_slice: np.ndarray | Any, pages: Sequence[PreparedPageTorch], *, trace: ExecutionTrace | None = None):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
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
        query = _pad_query(query_slice, header.head_dim, device_type=device_type)
        return torch.matmul(dense, query).reshape(-1)

    if header.mode_default == "M2":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        query_groups = query.reshape(header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
        for group_index in range(header.num_groups):
            group_sketch = torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0)
            group_basis = torch.stack([page.m2_basis[group_index] for page in pages], dim=0)
            group_mean = torch.stack([page.m2_mean[group_index] for page in pages], dim=0)
            if group_basis.dim() == 3:
                q_proj = torch.einsum("prg,g->pr", group_basis, query_groups[group_index])
                logits += torch.einsum("ptd,pd->pt", group_sketch, q_proj)
                logits += torch.einsum("pg,g->p", group_mean, query_groups[group_index])[:, None]
                continue
            segment_ids = torch.from_numpy(segment_ids_for_token_count(header.token_count, int(group_basis.shape[1]))).to(device=device_type)
            q_proj = torch.einsum("psrg,g->psr", group_basis, query_groups[group_index])
            logits += torch.einsum("ptr,ptr->pt", group_sketch, q_proj[:, segment_ids, :])
            logits += torch.einsum("ptg,g->pt", group_mean[:, segment_ids, :], query_groups[group_index])
        return logits.reshape(-1)

    if header.mode_default == "M1":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        query_groups = query.reshape(header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
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
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            logits += torch.matmul(group, query_groups[group_index])
        return logits.reshape(-1)

    if header.mode_default == "T3":
        query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
        rotated_query_groups = _fwht_last_dim_torch(query.reshape(header.num_groups, header.group_size))
        page_count = len(pages)
        logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            corrected = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            logits += torch.matmul(corrected, rotated_query_groups[group_index])
        return logits.reshape(-1)

    query = _pad_query(query_slice, header.padded_head_dim, device_type=device_type)
    query_groups = query.reshape(header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    page_count = len(pages)
    logits = torch.zeros((page_count, header.token_count), dtype=torch.float32, device=device_type)

    prepared_chunk = _get_prepared_chunk_mps(pages)
    for group_index in range(header.num_groups):
        codes = (
            prepared_chunk.codes_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.codes_groups is not None
            else _unpack_bits_torch(
                torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
        )
        if trace is not None and not (prepared_chunk is not None and prepared_chunk.codes_groups is not None):
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[group_index]
        int_dot = torch.matmul(codes, qg)
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        logits += scales * int_dot + bias * query_group_sums[group_index]

    return logits.reshape(-1)


def _mix_page_chunk_torch(
    attn_weights,
    pages: Sequence[PreparedPageTorch],
    *,
    out_acc=None,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    page_count = len(pages)
    token_count = header.token_count
    output = _prepare_output_accumulator_tensor(out_acc, header.head_dim, header.padded_head_dim, device_type=device_type)

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if not isinstance(attn_weights, torch.Tensor):
        weights = torch.as_tensor(attn_weights, dtype=torch.float32, device=device_type)
    else:
        weights = attn_weights.to(dtype=torch.float32, device=device_type)
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

    if header.mode_default == "M2":
        raise ValueError("M2 is only supported for key scoring in this phase")

    if header.mode_default == "M1":
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
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            start = group_index * header.group_size
            end = start + header.group_size
            output[start:end] += torch.einsum("pt,ptg->g", weights, group)
        return output

    if header.mode_default == "T3":
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            rotated_group = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            group = _fwht_last_dim_torch(rotated_group)
            start = group_index * header.group_size
            end = start + header.group_size
            output[start:end] += torch.einsum("pt,ptg->g", weights, group)
        return output

    prepared_chunk = _get_prepared_chunk_mps(pages)
    for group_index in range(header.num_groups):
        codes = (
            prepared_chunk.codes_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.codes_groups is not None
            else _unpack_bits_torch(
                torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
        )
        if trace is not None and not (prepared_chunk is not None and prepared_chunk.codes_groups is not None):
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        weighted_scales = weights * scales
        contribution = torch.sum(weighted_scales[..., None] * codes, dim=(0, 1))
        bias_term = torch.sum(weights * bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[start:end] += contribution + bias_term

    return output


def score_page_torch(
    query_slice: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    prepared = prepare_page_torch(page, device_type=device_type, trace=trace)
    return _score_page_chunk_torch(query_slice, [prepared], trace=trace).detach().cpu().numpy()


def score_pages_torch(
    query_slice: np.ndarray | Any,
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    prepared_pages = prepare_pages_torch(pages, device_type=device_type, trace=trace)
    if not prepared_pages:
        return []
    page_logits: list[np.ndarray] = []
    for page_chunk in _chunk_compatible_pages(prepared_pages):
        chunk_logits = _score_page_chunk_torch(query_slice, page_chunk, trace=trace)
        chunk_logits = chunk_logits.reshape(len(page_chunk), page_chunk[0].header.token_count)
        page_logits.extend(chunk_logits[index].detach().cpu().numpy() for index in range(len(page_chunk)))
    return page_logits


def mix_page_torch(
    attn_weights: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    device_type: TorchDevice,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    prepared = prepare_page_torch(page, device_type=device_type, trace=trace)
    header = prepared.header
    output = _mix_page_chunk_torch(
        np.asarray(attn_weights, dtype=np.float32)[None, :],
        [prepared],
        out_acc=None if out_acc is None else _prepare_output_accumulator(out_acc, header.head_dim, header.padded_head_dim, device_type=device_type),
        trace=trace,
    )
    return output[: header.head_dim].detach().cpu().numpy()


def _page_logits_tensor(page_logits, token_count: int, *, device_type: TorchDevice):
    torch = _load_torch()
    logits = torch.as_tensor(page_logits, dtype=torch.float32, device=device_type)
    if tuple(logits.shape) != (token_count,):
        raise ValueError("precomputed page logits must have shape [token_count]")
    return logits


def decode_step_torch(
    query_slice: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    precomputed_page_logits: Sequence[np.ndarray | Any | None] | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = _load_torch()
    prepared_key_pages = prepare_pages_torch(key_pages, device_type=device_type, trace=trace)
    prepared_value_pages = prepare_pages_torch(value_pages, device_type=device_type, trace=trace)
    if not prepared_key_pages:
        raise ValueError(f"decode_step_{device_type} requires at least one page")
    if precomputed_page_logits is not None and len(precomputed_page_logits) != len(prepared_key_pages):
        raise ValueError("precomputed_page_logits must align with key_pages")

    logits_parts = []
    score_run: list[PreparedPageTorch] = []

    def flush_score_run() -> None:
        nonlocal score_run
        if not score_run:
            return
        chunk_logits = _score_page_chunk_torch(query_slice, score_run, trace=trace)
        chunk_logits = chunk_logits.reshape(len(score_run), score_run[0].header.token_count)
        logits_parts.extend(chunk_logits[index] for index in range(len(score_run)))
        score_run = []

    for index, page in enumerate(prepared_key_pages):
        cached_logits = None if precomputed_page_logits is None else precomputed_page_logits[index]
        if cached_logits is not None:
            flush_score_run()
            logits_parts.append(_page_logits_tensor(cached_logits, page.header.token_count, device_type=device_type))
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
        device=device_type,
    )
    offset = 0
    for page_chunk in _chunk_compatible_pages(prepared_value_pages):
        chunk_token_count = page_chunk[0].header.token_count * len(page_chunk)
        chunk_weights = weights[offset : offset + chunk_token_count].reshape(len(page_chunk), page_chunk[0].header.token_count)
        output = _mix_page_chunk_torch(chunk_weights, page_chunk, out_acc=output, trace=trace)
        offset += chunk_token_count

    head_dim = prepared_value_pages[0].header.head_dim
    return (
        logits.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        output[:head_dim].detach().cpu().numpy(),
    )


def _score_page_chunk_multiquery_torch(
    query_slices: np.ndarray | Any,
    pages: Sequence[PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
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
        queries = _pad_queries(query_slices, header.head_dim, device_type=device_type)
        return torch.einsum("pth,qh->qpt", dense, queries).reshape(query_count, -1)

    if header.mode_default == "M2":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        for group_index in range(header.num_groups):
            group_sketch = torch.stack([page.m2_sketch[:, group_index, :] for page in pages], dim=0)
            group_basis = torch.stack([page.m2_basis[group_index] for page in pages], dim=0)
            group_mean = torch.stack([page.m2_mean[group_index] for page in pages], dim=0)
            if group_basis.dim() == 3:
                q_proj = torch.einsum("prg,qg->qpr", group_basis, query_groups[:, group_index, :])
                logits += torch.einsum("ptd,qpd->qpt", group_sketch, q_proj)
                logits += torch.einsum("pg,qg->qp", group_mean, query_groups[:, group_index, :])[:, :, None]
                continue
            segment_ids = torch.from_numpy(segment_ids_for_token_count(header.token_count, int(group_basis.shape[1]))).to(device=device_type)
            q_proj = torch.einsum("psrg,qg->qpsr", group_basis, query_groups[:, group_index, :])
            logits += torch.einsum("ptr,qptr->qpt", group_sketch, q_proj[:, :, segment_ids, :])
            logits += torch.einsum("ptg,qg->qpt", group_mean[:, segment_ids, :], query_groups[:, group_index, :])
        return logits.reshape(query_count, -1)

    if header.mode_default == "M1":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
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
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            logits += torch.einsum("ptg,qg->qpt", group, query_groups[:, group_index, :])
        return logits.reshape(query_count, -1)

    if header.mode_default == "T3":
        queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
        rotated_query_groups = _fwht_last_dim_torch(queries.reshape(query_count, header.num_groups, header.group_size))
        page_count = len(pages)
        logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            corrected = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            logits += torch.einsum("ptg,qg->qpt", corrected, rotated_query_groups[:, group_index, :])
        return logits.reshape(query_count, -1)

    queries = _pad_queries(query_slices, header.padded_head_dim, device_type=device_type)
    query_groups = queries.reshape(query_count, header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(dim=-1)
    page_count = len(pages)
    logits = torch.zeros((query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)

    prepared_chunk = _get_prepared_chunk_mps(pages)
    for group_index in range(header.num_groups):
        codes = (
            prepared_chunk.codes_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.codes_groups is not None
            else _unpack_bits_torch(
                torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, header.token_count, header.group_size)
        )
        if trace is not None and not (prepared_chunk is not None and prepared_chunk.codes_groups is not None):
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups[:, group_index, :]
        int_dot = torch.einsum("ptg,qg->qpt", codes, qg)
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        logits += scales.unsqueeze(0) * int_dot + bias.unsqueeze(0) * query_group_sums[:, group_index].reshape(
            query_count,
            1,
            1,
        )

    return logits.reshape(query_count, -1)


def _mix_page_chunk_multiquery_torch(
    attn_weights,
    pages: Sequence[PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages:
        raise ValueError("pages must be non-empty")
    header = pages[0].header
    device_type = pages[0].device_type
    page_count = len(pages)
    token_count = header.token_count

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for page in pages),
            sum(page.metadata_nbytes for page in pages),
        )

    if not isinstance(attn_weights, torch.Tensor):
        weights = torch.as_tensor(attn_weights, dtype=torch.float32, device=device_type)
    else:
        weights = attn_weights.to(dtype=torch.float32, device=device_type)
    if weights.ndim != 3 or tuple(weights.shape[1:]) != (page_count, token_count):
        raise ValueError("attn_weights chunk must have shape [query_count, page_count, token_count]")

    query_count = int(weights.shape[0])
    output = torch.zeros((query_count, header.padded_head_dim), dtype=torch.float32, device=device_type)

    if header.mode_default == "M3":
        dense = torch.stack(
            [page.escape_payload[: header.token_count, : header.head_dim].to(torch.float32) for page in pages],
            dim=0,
        )
        output[:, : header.head_dim] += torch.einsum("qpt,pth->qh", weights, dense)
        return output

    if header.mode_default == "M2":
        raise ValueError("M2 is only supported for key scoring in this phase")

    if header.mode_default == "M1":
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
            group = _lookup_lut_group_torch(
                torch.stack([page.codebooks[group_index] for page in pages], dim=0),
                codes,
            )
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, start:end] += torch.einsum("qpt,ptg->qg", weights, group)
        return output

    if header.mode_default == "T3":
        codebooks = pages[0].codebooks if pages[0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunk = _get_prepared_chunk_mps(pages)
        for group_index in range(header.num_groups):
            group_words = (
                prepared_chunk.payload_groups[group_index]
                if prepared_chunk is not None
                else torch.stack([page.payload[group_index] for page in pages], dim=0)
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = (
                prepared_chunk.scales_groups[group_index]
                if prepared_chunk is not None and prepared_chunk.scales_groups is not None
                else torch.stack([page.scales[:, group_index] for page in pages], dim=0)
            )
            rotated_group = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            group = _fwht_last_dim_torch(rotated_group)
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, start:end] += torch.einsum("qpt,ptg->qg", weights, group)
        return output

    prepared_chunk = _get_prepared_chunk_mps(pages)
    for group_index in range(header.num_groups):
        codes = (
            prepared_chunk.codes_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.codes_groups is not None
            else _unpack_bits_torch(
                torch.stack([page.payload[group_index] for page in pages], dim=0).reshape(-1, header.words_per_group),
                pages[0].unpack_shifts,
                pages[0].unpack_mask,
                header.group_size,
            ).reshape(page_count, token_count, header.group_size)
        )
        if trace is not None and not (prepared_chunk is not None and prepared_chunk.codes_groups is not None):
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = (
            prepared_chunk.scales_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.scales_groups is not None
            else torch.stack([page.scales[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        bias = (
            prepared_chunk.bias_groups[group_index]
            if prepared_chunk is not None and prepared_chunk.bias_groups is not None
            else torch.stack([page.bias[:, group_index].to(torch.float32) for page in pages], dim=0)
        )
        weighted_scales = weights * scales.unsqueeze(0)
        contribution = torch.einsum("qpt,ptg->qg", weighted_scales, codes)
        bias_term = torch.einsum("qpt,pt->q", weights, bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[:, start:end] += contribution + bias_term[:, None]

    return output


def _score_page_chunk_grouped_multiquery_torch(
    query_groups,
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages_by_group or not pages_by_group[0]:
        raise ValueError("pages_by_group must be non-empty")
    batch_size = len(pages_by_group)
    page_count = len(pages_by_group[0])
    header = pages_by_group[0][0].header
    device_type = pages_by_group[0][0].device_type

    for group_pages in pages_by_group:
        if len(group_pages) != page_count:
            raise ValueError("all page groups must have the same page count")
        for page in group_pages:
            if _batched_signature(page) != _batched_signature(pages_by_group[0][0]):
                raise ValueError("all grouped pages must share the same page signature within a chunk")

    queries = torch.stack(
        [
            group.to(dtype=torch.float32, device=device_type)
            if torch.is_tensor(group)
            else torch.as_tensor(group, dtype=torch.float32, device=device_type)
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

    if header.mode_default == "M2":
        padded_queries = _pad_queries(
            queries.reshape(batch_size * query_count, header.head_dim),
            header.padded_head_dim,
            device_type=device_type,
        ).reshape(batch_size, query_count, header.padded_head_dim)
        query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        for group_index in range(header.num_groups):
            group_sketch = torch.stack(
                [torch.stack([page.m2_sketch[:, group_index, :] for page in group_pages], dim=0) for group_pages in pages_by_group],
                dim=0,
            )
            group_basis = torch.stack(
                [torch.stack([page.m2_basis[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                dim=0,
            )
            group_mean = torch.stack(
                [torch.stack([page.m2_mean[group_index] for page in group_pages], dim=0) for group_pages in pages_by_group],
                dim=0,
            )
            if group_basis.dim() == 4:
                q_proj = torch.einsum("bprg,bqg->bqpr", group_basis, query_groups_tensor[:, :, group_index, :])
                logits += torch.einsum("bptd,bqpd->bqpt", group_sketch, q_proj)
                logits += torch.einsum("bpg,bqg->bqp", group_mean, query_groups_tensor[:, :, group_index, :])[:, :, :, None]
                continue
            segment_ids = torch.from_numpy(segment_ids_for_token_count(header.token_count, int(group_basis.shape[2]))).to(device=device_type)
            q_proj = torch.einsum("bpsrg,bqg->bqpsr", group_basis, query_groups_tensor[:, :, group_index, :])
            logits += torch.einsum("bptr,bqptr->bqpt", group_sketch, q_proj[:, :, :, segment_ids, :])
            logits += torch.einsum("bptg,bqg->bqpt", group_mean[:, :, segment_ids, :], query_groups_tensor[:, :, group_index, :])
        return logits.reshape(batch_size, query_count, -1)

    if header.mode_default == "M1":
        padded_queries = _pad_queries(
            queries.reshape(batch_size * query_count, header.head_dim),
            header.padded_head_dim,
            device_type=device_type,
        ).reshape(batch_size, query_count, header.padded_head_dim)
        query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
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
                trace=trace,
            ).reshape(batch_size, page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack(
                    [
                        torch.stack([page.codebooks[group_index] for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                ),
                codes,
            )
            logits += torch.einsum("bptg,bqg->bqpt", group, query_groups_tensor[:, :, group_index, :])
        return logits.reshape(batch_size, query_count, -1)

    if header.mode_default == "T3":
        padded_queries = _pad_queries(
            queries.reshape(batch_size * query_count, header.head_dim),
            header.padded_head_dim,
            device_type=device_type,
        ).reshape(batch_size, query_count, header.padded_head_dim)
        rotated_query_groups = _fwht_last_dim_torch(
            padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size),
            trace=trace,
        )
        logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)
        codebooks = pages_by_group[0][0].codebooks if pages_by_group[0][0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunks = [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
        for group_index in range(header.num_groups):
            group_words = torch.stack(
                [
                    prepared_chunks[group_id].payload_groups[group_index]
                    if prepared_chunks[group_id] is not None
                    else torch.stack([page.payload[group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages_by_group[0][0].unpack_shifts,
                pages_by_group[0][0].unpack_mask,
                header.group_size,
                trace=trace,
            ).reshape(batch_size, page_count, header.token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            corrected = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            logits += torch.einsum("bptg,bqg->bqpt", corrected, rotated_query_groups[:, :, group_index, :])
        return logits.reshape(batch_size, query_count, -1)

    padded_queries = _pad_queries(
        queries.reshape(batch_size * query_count, header.head_dim),
        header.padded_head_dim,
        device_type=device_type,
    ).reshape(batch_size, query_count, header.padded_head_dim)
    query_groups_tensor = padded_queries.reshape(batch_size, query_count, header.num_groups, header.group_size)
    query_group_sums = query_groups_tensor.sum(dim=-1)
    logits = torch.zeros((batch_size, query_count, page_count, header.token_count), dtype=torch.float32, device=device_type)

    grouped_prepared_chunk = _get_grouped_prepared_chunk_mps(pages_by_group)
    prepared_chunks = None if grouped_prepared_chunk is not None else [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
    for group_index in range(header.num_groups):
        codes = (
            grouped_prepared_chunk.codes_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.codes_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].codes_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].codes_groups is not None
                    else _unpack_bits_torch(
                        torch.stack([page.payload[group_index] for page in group_pages], dim=0).reshape(-1, header.words_per_group),
                        pages_by_group[0][0].unpack_shifts,
                        pages_by_group[0][0].unpack_mask,
                        header.group_size,
                        trace=trace,
                    ).reshape(page_count, header.token_count, header.group_size)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        if trace is not None and not (grouped_prepared_chunk is not None and grouped_prepared_chunk.codes_groups is not None):
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        qg = query_groups_tensor[:, :, group_index, :]
        int_dot = torch.einsum("bptg,bqg->bqpt", codes, qg)
        scales = (
            grouped_prepared_chunk.scales_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.scales_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        bias = (
            grouped_prepared_chunk.bias_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.bias_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].bias_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].bias_groups is not None
                    else torch.stack([page.bias[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        logits += scales[:, None] * int_dot + bias[:, None] * query_group_sums[:, :, group_index].reshape(
            batch_size,
            query_count,
            1,
            1,
        )

    return logits.reshape(batch_size, query_count, -1)


def _mix_page_chunk_grouped_multiquery_torch(
    attn_weights,
    pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    if not pages_by_group or not pages_by_group[0]:
        raise ValueError("pages_by_group must be non-empty")
    batch_size = len(pages_by_group)
    page_count = len(pages_by_group[0])
    header = pages_by_group[0][0].header
    device_type = pages_by_group[0][0].device_type
    token_count = header.token_count

    if trace is not None:
        trace.record_page_read(
            sum(page.payload_nbytes for group_pages in pages_by_group for page in group_pages),
            sum(page.metadata_nbytes for group_pages in pages_by_group for page in group_pages),
        )

    weights = attn_weights if isinstance(attn_weights, torch.Tensor) else torch.as_tensor(attn_weights, dtype=torch.float32, device=device_type)
    weights = weights.to(dtype=torch.float32, device=device_type)
    if weights.ndim != 4 or tuple(weights.shape[2:]) != (page_count, token_count):
        raise ValueError("grouped attn_weights chunk must have shape [batch_size, query_count, page_count, token_count]")

    query_count = int(weights.shape[1])
    output = torch.zeros((batch_size, query_count, header.padded_head_dim), dtype=torch.float32, device=device_type)

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

    if header.mode_default == "M2":
        raise ValueError("M2 is only supported for key scoring in this phase")

    if header.mode_default == "M1":
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
                trace=trace,
            ).reshape(batch_size, page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            group = _lookup_lut_group_torch(
                torch.stack(
                    [
                        torch.stack([page.codebooks[group_index] for page in group_pages], dim=0)
                        for group_pages in pages_by_group
                    ],
                    dim=0,
                ),
                codes,
            )
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, :, start:end] += torch.einsum("bqpt,bptg->bqg", weights, group)
        return output

    if header.mode_default == "T3":
        codebooks = pages_by_group[0][0].codebooks if pages_by_group[0][0].codebooks is not None else _turbo3_centroids_torch(device_type=device_type)
        prepared_chunks = [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
        for group_index in range(header.num_groups):
            group_words = torch.stack(
                [
                    prepared_chunks[group_id].payload_groups[group_index]
                    if prepared_chunks[group_id] is not None
                    else torch.stack([page.payload[group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            codes = _unpack_bits_torch(
                group_words.reshape(-1, header.words_per_group),
                pages_by_group[0][0].unpack_shifts,
                pages_by_group[0][0].unpack_mask,
                header.group_size,
                trace=trace,
            ).reshape(batch_size, page_count, token_count, header.group_size)
            if trace is not None:
                trace.record_temporary(int(codes.numel() * codes.element_size()))
            scales = torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index] for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
            rotated_group = _lookup_turbo_group_torch(codebooks, codes) * scales[..., None]
            group = _fwht_last_dim_torch(rotated_group, trace=trace)
            start = group_index * header.group_size
            end = start + header.group_size
            output[:, :, start:end] += torch.einsum("bqpt,bptg->bqg", weights, group)
        return output

    grouped_prepared_chunk = _get_grouped_prepared_chunk_mps(pages_by_group)
    prepared_chunks = None if grouped_prepared_chunk is not None else [_get_prepared_chunk_mps(group_pages) for group_pages in pages_by_group]
    for group_index in range(header.num_groups):
        codes = (
            grouped_prepared_chunk.codes_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.codes_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].codes_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].codes_groups is not None
                    else _unpack_bits_torch(
                        torch.stack([page.payload[group_index] for page in group_pages], dim=0).reshape(-1, header.words_per_group),
                        pages_by_group[0][0].unpack_shifts,
                        pages_by_group[0][0].unpack_mask,
                        header.group_size,
                        trace=trace,
                    ).reshape(page_count, token_count, header.group_size)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        if trace is not None and not (grouped_prepared_chunk is not None and grouped_prepared_chunk.codes_groups is not None):
            trace.record_temporary(int(codes.numel() * codes.element_size()))
        scales = (
            grouped_prepared_chunk.scales_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.scales_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].scales_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].scales_groups is not None
                    else torch.stack([page.scales[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        bias = (
            grouped_prepared_chunk.bias_groups[group_index]
            if grouped_prepared_chunk is not None and grouped_prepared_chunk.bias_groups is not None
            else torch.stack(
                [
                    prepared_chunks[group_id].bias_groups[group_index]
                    if prepared_chunks is not None and prepared_chunks[group_id] is not None and prepared_chunks[group_id].bias_groups is not None
                    else torch.stack([page.bias[:, group_index].to(torch.float32) for page in group_pages], dim=0)
                    for group_id, group_pages in enumerate(pages_by_group)
                ],
                dim=0,
            )
        )
        weighted_scales = weights * scales[:, None]
        contribution = torch.einsum("bqpt,bptg->bqg", weighted_scales, codes)
        bias_term = torch.einsum("bqpt,bpt->bq", weights, bias)
        start = group_index * header.group_size
        end = start + header.group_size
        output[:, :, start:end] += contribution + bias_term[:, :, None]

    return output


def decode_multi_query_step_torch_tensor(
    query_slices: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
):
    torch = _load_torch()
    prepared_key_pages = prepare_pages_torch(key_pages, device_type=device_type, trace=trace)
    prepared_value_pages = prepare_pages_torch(value_pages, device_type=device_type, trace=trace)
    if not prepared_key_pages:
        raise ValueError(f"decode_multi_query_step_{device_type} requires at least one page")

    logits_parts = []
    for page_chunk in _chunk_compatible_pages(prepared_key_pages):
        logits_parts.append(_score_page_chunk_multiquery_torch(query_slices, page_chunk, trace=trace))
    logits = torch.cat(logits_parts, dim=1)
    weights = torch.softmax(logits, dim=1)

    if torch.is_tensor(query_slices):
        query_count = int(query_slices.shape[0])
    else:
        query_count = int(np.asarray(query_slices).shape[0])
    output = torch.zeros(
        (query_count, prepared_value_pages[0].header.padded_head_dim),
        dtype=torch.float32,
        device=device_type,
    )
    offset = 0
    for page_chunk in _chunk_compatible_pages(prepared_value_pages):
        chunk_token_count = page_chunk[0].header.token_count * len(page_chunk)
        chunk_weights = weights[:, offset : offset + chunk_token_count].reshape(
            weights.shape[0],
            len(page_chunk),
            page_chunk[0].header.token_count,
        )
        output += _mix_page_chunk_multiquery_torch(chunk_weights, page_chunk, trace=trace)
        offset += chunk_token_count

    head_dim = prepared_value_pages[0].header.head_dim
    return logits, weights, output[:, :head_dim]


def decode_grouped_multiquery_step_torch_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
):
    if not key_pages_by_group or not value_pages_by_group:
        raise ValueError("grouped decode requires non-empty key/value page groups")
    if len(key_pages_by_group) != len(value_pages_by_group):
        raise ValueError("key_pages_by_group and value_pages_by_group must have the same group count")
    group_count = len(key_pages_by_group)
    if len(query_groups) != group_count:
        raise ValueError("query_groups must align with key/value group count")

    prepared_key_groups = [prepare_pages_torch(group_pages, device_type=device_type, trace=trace) for group_pages in key_pages_by_group]
    prepared_value_groups = [prepare_pages_torch(group_pages, device_type=device_type, trace=trace) for group_pages in value_pages_by_group]
    if not prepared_key_groups[0]:
        raise ValueError("grouped decode requires at least one key page per group")

    return decode_grouped_multiquery_step_prepared_torch_tensor(
        query_groups,
        prepared_key_groups,
        prepared_value_groups,
        trace=trace,
    )


def decode_grouped_multiquery_step_prepared_torch_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
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
    device_type = key_pages_by_group[0][0].device_type

    query_tensors = [
        group.to(dtype=torch.float32, device=device_type)
        if torch.is_tensor(group)
        else torch.as_tensor(group, dtype=torch.float32, device=device_type)
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
        logits_parts.append(
            _trace_timed_call(
                trace,
                "score",
                device_type=device_type,
                fn=lambda chunk_pages=chunk_pages: _score_page_chunk_grouped_multiquery_torch(
                    query_tensors,
                    chunk_pages,
                    trace=trace,
                ),
            )
        )
        key_offset += chunk_length
    logits = torch.cat(logits_parts, dim=2)
    weights = _trace_timed_call(
        trace,
        "softmax",
        device_type=device_type,
        fn=lambda: torch.softmax(logits, dim=2),
    )

    head_dim = first_value_group[0].header.head_dim
    padded_head_dim = first_value_group[0].header.padded_head_dim
    output = torch.zeros((group_count, query_count, padded_head_dim), dtype=torch.float32, device=device_type)
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
        output += _trace_timed_call(
            trace,
            "mix",
            device_type=device_type,
            fn=lambda chunk_weights=chunk_weights, chunk_pages=chunk_pages: _mix_page_chunk_grouped_multiquery_torch(
                chunk_weights,
                chunk_pages,
                trace=trace,
            ),
        )
        offset += chunk_token_count
        value_offset += chunk_length

    return logits, weights, output[:, :, :head_dim]


def decode_multi_query_step_torch(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    device_type: TorchDevice,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits, weights, output = decode_multi_query_step_torch_tensor(
        query_slices,
        key_pages,
        value_pages,
        device_type=device_type,
        trace=trace,
    )
    return (
        logits.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        output.detach().cpu().numpy(),
    )


def prepare_pages_mps(
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> list[PreparedPageTorch]:
    return prepare_pages_torch(pages, device_type="mps", trace=trace)


def prepare_page_mps(page: EncodedPage | PreparedPageTorch, *, trace: ExecutionTrace | None = None) -> PreparedPageTorch:
    return prepare_page_torch(page, device_type="mps", trace=trace)


def score_page_mps(
    query_slice: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    return score_page_torch(query_slice, page, device_type="mps", trace=trace)


def score_pages_mps(
    query_slice: np.ndarray | Any,
    pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> list[np.ndarray]:
    return score_pages_torch(query_slice, pages, device_type="mps", trace=trace)


def mix_page_mps(
    attn_weights: np.ndarray | Any,
    page: EncodedPage | PreparedPageTorch,
    *,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    return mix_page_torch(attn_weights, page, device_type="mps", out_acc=out_acc, trace=trace)


def decode_step_mps(
    query_slice: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    precomputed_page_logits: Sequence[np.ndarray | Any | None] | None = None,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_step_torch(
        query_slice,
        key_pages,
        value_pages,
        device_type="mps",
        precomputed_page_logits=precomputed_page_logits,
        trace=trace,
    )


def decode_multi_query_step_mps_tensor(
    query_slices: np.ndarray | Any,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_multi_query_step_torch_tensor(
        query_slices,
        key_pages,
        value_pages,
        device_type="mps",
        trace=trace,
    )


def decode_grouped_multiquery_step_mps_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[EncodedPage | PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_grouped_multiquery_step_torch_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
        device_type="mps",
        trace=trace,
    )


def decode_grouped_multiquery_step_prepared_mps_tensor(
    query_groups,
    key_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    value_pages_by_group: Sequence[Sequence[PreparedPageTorch]],
    *,
    trace: ExecutionTrace | None = None,
):
    return decode_grouped_multiquery_step_prepared_torch_tensor(
        query_groups,
        key_pages_by_group,
        value_pages_by_group,
        trace=trace,
    )


def decode_multi_query_step_mps(
    query_slices: np.ndarray,
    key_pages: Sequence[EncodedPage | PreparedPageTorch],
    value_pages: Sequence[EncodedPage | PreparedPageTorch],
    *,
    trace: ExecutionTrace | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return decode_multi_query_step_torch(query_slices, key_pages, value_pages, device_type="mps", trace=trace)
