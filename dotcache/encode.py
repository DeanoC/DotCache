from __future__ import annotations

import numpy as np

from .config import DotCacheConfig
from .modes.m0_affine import quantize_tensor
from .modes.m1_lut import quantize_tensor_lut
from .modes.m3_escape import encode_escape_payload
from .page_format import build_payload
from .packing import words_per_group
from .types import EncodedPage, Kind, PageHeader

DEFAULT_RUNTIME_SKETCH_ROWS = 4


def _build_runtime_page_sketch(values: np.ndarray, *, sketch_rows: int = DEFAULT_RUNTIME_SKETCH_ROWS) -> tuple[np.ndarray, np.ndarray]:
    rows = min(max(1, sketch_rows), values.shape[0])
    chunks = np.array_split(values, rows, axis=0)
    sketch = np.stack([chunk.mean(axis=0) for chunk in chunks], axis=0).astype(np.float16)
    page_mean = values.mean(axis=0).astype(np.float16)
    return page_mean, sketch


def _build_runtime_page_envelope(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    page_min = values.min(axis=0).astype(np.float16)
    page_max = values.max(axis=0).astype(np.float16)
    return page_min, page_max


def encode_page(
    tensor_slice: np.ndarray,
    config: DotCacheConfig,
    *,
    kind: Kind,
    layer_id: int = 0,
    kv_head_id: int = 0,
    token_start: int = 0,
    mode: str | None = None,
    layout: str | None = None,
    quant_scheme: str | None = None,
    build_runtime_metadata: bool = True,
) -> EncodedPage:
    values = np.asarray(tensor_slice, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("tensor_slice must have shape [token_count, head_dim]")
    if values.shape[1] != config.head_dim:
        raise ValueError("tensor_slice head_dim must match config.head_dim")

    bits = config.bits_k if kind == "K" else config.bits_v
    default_mode = config.default_mode_k if kind == "K" else config.default_mode_v
    page_mode = mode or default_mode
    page_layout = layout or (config.payload_layout_k if kind == "K" else config.payload_layout_v)
    scheme = quant_scheme or (config.quant_scheme_k if kind == "K" else config.quant_scheme_v)
    token_count = values.shape[0]
    runtime_page_mean = None
    runtime_page_sketch = None
    runtime_page_min = None
    runtime_page_max = None
    if build_runtime_metadata:
        runtime_page_mean, runtime_page_sketch = _build_runtime_page_sketch(values)
        runtime_page_min, runtime_page_max = _build_runtime_page_envelope(values)

    if page_mode == "M3":
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=token_start,
            token_count=token_count,
            head_dim=config.head_dim,
            padded_head_dim=config.padded_head_dim,
            group_size=config.group_size,
            num_groups=config.num_groups,
            bits=bits,
            words_per_group=words_per_group(config.group_size, bits),
            mode_default="M3",
            layout=page_layout,
            quant_scheme=scheme,
            escape_dtype=config.escape_dtype,
        )
        escape_payload = encode_escape_payload(values, dtype=config.escape_dtype)
        return EncodedPage(
            header=header,
            escape_payload=escape_payload,
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    if page_mode == "M1":
        codes, codebooks, padded_head_dim = quantize_tensor_lut(
            values,
            group_size=config.group_size,
            bits=bits,
            refine_steps=config.lut_refine_steps,
            preconditioner=config.preconditioner,
            precondition_strength=config.precondition_strength,
        )
        payload = build_payload(codes, bits, page_layout)
        header = PageHeader(
            layer_id=layer_id,
            kv_head_id=kv_head_id,
            kind=kind,
            token_start=token_start,
            token_count=token_count,
            head_dim=config.head_dim,
            padded_head_dim=padded_head_dim,
            group_size=config.group_size,
            num_groups=config.num_groups,
            bits=bits,
            words_per_group=words_per_group(config.group_size, bits),
            mode_default="M1",
            layout=page_layout,
            quant_scheme="lut",
            escape_dtype=config.escape_dtype,
        )
        return EncodedPage(
            header=header,
            payload=payload,
            codebooks=codebooks.astype(np.float16),
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    if page_mode != "M0":
        raise ValueError("only M0, M1, and M3 are supported in this bootstrap")

    codes, scales, bias, padded_head_dim = quantize_tensor(
        values,
        group_size=config.group_size,
        bits=bits,
        scheme=scheme,
    )
    payload = build_payload(codes, bits, page_layout)
    header = PageHeader(
        layer_id=layer_id,
        kv_head_id=kv_head_id,
        kind=kind,
        token_start=token_start,
        token_count=token_count,
        head_dim=config.head_dim,
        padded_head_dim=padded_head_dim,
        group_size=config.group_size,
        num_groups=config.num_groups,
        bits=bits,
        words_per_group=words_per_group(config.group_size, bits),
        mode_default="M0",
        layout=page_layout,
        quant_scheme=scheme,
        escape_dtype=config.escape_dtype,
    )
    stored_scales = scales.astype(np.float16)
    stored_bias = None if bias is None else bias.astype(np.float16)
    return EncodedPage(
        header=header,
        payload=payload,
        scales=stored_scales,
        bias=stored_bias,
        runtime_page_mean=runtime_page_mean,
        runtime_page_sketch=runtime_page_sketch,
        runtime_page_min=runtime_page_min,
        runtime_page_max=runtime_page_max,
    )
