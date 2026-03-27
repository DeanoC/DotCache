from __future__ import annotations

import numpy as np

from .config import DotCacheConfig
from .modes.m0_affine import quantize_tensor
from .modes.m1_lut import quantize_tensor_lut
from .modes.m2_key_sketch import quantize_tensor_m2, reconstruct_group_m2
from .modes.m3_escape import encode_escape_payload
from .page_format import build_payload
from .packing import words_per_group
from .types import EncodedPage, Kind, PageHeader

DEFAULT_RUNTIME_SKETCH_ROWS = 4


def _reconstruct_lut_page(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    token_count, num_groups, group_size = codes.shape
    dense = np.zeros((token_count, num_groups * group_size), dtype=np.float32)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        group_codebook = codebooks[group_index].astype(np.float32)
        if group_codebook.ndim == 1:
            dense[:, start:end] = group_codebook[codes[:, group_index].astype(np.int64)]
        else:
            segment_count = group_codebook.shape[0]
            segment_ids = (np.arange(token_count, dtype=np.int64) * segment_count) // max(token_count, 1)
            dense[:, start:end] = group_codebook[segment_ids[:, None], codes[:, group_index].astype(np.int64)]
    return dense


def _reconstruct_m2_page(coeffs: np.ndarray, basis: np.ndarray, *, group_size: int) -> np.ndarray:
    token_count, num_groups, _ = coeffs.shape
    dense = np.zeros((token_count, num_groups * group_size), dtype=np.float32)
    for group_index in range(num_groups):
        start = group_index * group_size
        end = start + group_size
        dense[:, start:end] = reconstruct_group_m2(coeffs[:, group_index, :], basis=basis[group_index])
    return dense


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
    build_m2_sidecar: bool | None = None,
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
    requested_mode = page_mode
    trial_quant_error = None
    runtime_page_mean = None
    runtime_page_sketch = None
    runtime_page_min = None
    runtime_page_max = None
    if build_runtime_metadata:
        runtime_page_mean, runtime_page_sketch = _build_runtime_page_sketch(values)
        runtime_page_min, runtime_page_max = _build_runtime_page_envelope(values)

    def _build_m2_sidecar() -> tuple[np.ndarray | None, np.ndarray | None]:
        sidecar_enabled = config.m2_prefilter_top_k > 0 if build_m2_sidecar is None else bool(build_m2_sidecar)
        if kind != "K" or not sidecar_enabled:
            return None, None
        coeffs, basis, _ = quantize_tensor_m2(
            values,
            group_size=config.group_size,
            sketch_dim=config.m2_sketch_dim_k,
        )
        return coeffs.astype(np.float16, copy=False), basis.astype(np.float16, copy=False)

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
            requested_mode=page_mode,
            runtime_page_mean=runtime_page_mean,
            runtime_page_sketch=runtime_page_sketch,
            runtime_page_min=runtime_page_min,
            runtime_page_max=runtime_page_max,
        )

    trial_token_p95_error = None

    if page_mode == "M2":
        if kind != "K":
            raise ValueError("M2 is only supported for K pages in this phase")
        coeffs, basis, padded_head_dim = quantize_tensor_m2(
            values,
            group_size=config.group_size,
            sketch_dim=config.m2_sketch_dim_k,
        )
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
            words_per_group=0,
            mode_default="M2",
            layout=page_layout,
            quant_scheme="sketch",
            escape_dtype=config.escape_dtype,
        )
        return EncodedPage(
            header=header,
            m2_sketch=coeffs.astype(np.float16, copy=False),
            m2_basis=basis.astype(np.float16, copy=False),
            requested_mode=page_mode,
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
            segment_count=config.m1_segment_count_k if kind == "K" else config.m1_segment_count_v,
            refine_steps=config.lut_refine_steps,
            preconditioner=config.preconditioner,
            precondition_strength=config.precondition_strength,
        )
        if config.m1_fallback_to_m0:
            reconstructed = _reconstruct_lut_page(codes, codebooks)[:, : config.head_dim]
            rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
            trial_quant_error = float(np.mean(np.abs(values - reconstructed), dtype=np.float64) / max(rms, 1e-6))
            token_norms = np.linalg.norm(values, axis=1)
            token_rel_error = np.linalg.norm(values - reconstructed, axis=1) / np.maximum(token_norms, 1e-6)
            trial_token_p95_error = float(np.percentile(token_rel_error, 95))
            if (
                trial_quant_error > config.m1_error_threshold
                or trial_token_p95_error > config.m1_token_p95_error_threshold
            ):
                page_mode = "M0"
                scheme = "affine"
        if page_mode == "M1":
            sidecar_sketch, sidecar_basis = _build_m2_sidecar()
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
                m2_sketch=sidecar_sketch,
                m2_basis=sidecar_basis,
                lut_segment_count=int(codebooks.shape[1]) if codebooks.ndim == 3 else 1,
                requested_mode=requested_mode,
                trial_quant_error=trial_quant_error,
                trial_token_p95_error=trial_token_p95_error,
                runtime_page_mean=runtime_page_mean,
                runtime_page_sketch=runtime_page_sketch,
                runtime_page_min=runtime_page_min,
                runtime_page_max=runtime_page_max,
            )

    if page_mode != "M0":
        raise ValueError("only M0, M1, M2, and M3 are supported in this bootstrap")

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
    sidecar_sketch, sidecar_basis = _build_m2_sidecar()
    return EncodedPage(
        header=header,
        payload=payload,
        scales=stored_scales,
        bias=stored_bias,
        m2_sketch=sidecar_sketch,
        m2_basis=sidecar_basis,
        requested_mode=requested_mode,
        trial_quant_error=trial_quant_error,
        trial_token_p95_error=trial_token_p95_error if "trial_token_p95_error" in locals() else None,
        runtime_page_mean=runtime_page_mean,
        runtime_page_sketch=runtime_page_sketch,
        runtime_page_min=runtime_page_min,
        runtime_page_max=runtime_page_max,
    )
